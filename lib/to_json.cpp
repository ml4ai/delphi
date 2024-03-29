#include "AnalysisGraph.hpp"
#include <fmt/format.h>

using namespace std;
using namespace fmt::literals;
using namespace nlohmann;

typedef vector<pair<tuple<string, int, string>, tuple<string, int, string>>>
Evidence;

// TODO: For debugging remove later.
using fmt::print;

string AnalysisGraph::to_json_string(int indent) {
  nlohmann::json j;
  j["id"] = this->id;
  j["experiment_id"] = this->experiment_id;
  j["edges"] = {};
  vector<tuple<string, string, vector<double>>> data;
  for (auto e : this->edges()) {
    string source = (*this)[boost::source(e, this->graph)].name;
    string target = (*this)[boost::target(e, this->graph)].name;
    j["edges"].push_back({
        {"source", source},
        {"target", target},
        {"kernels", this->edge(e).kde.dataset},
    });
  }
  for (Node& n : this->nodes()) {
    // Just taking the first indicator for now, will try to incorporate multiple
    // indicators per node later.
    if (n.indicators.size() == 0) {
      throw runtime_error("Node {} has no indicators!"_format(n.name));
    }
    Indicator& indicator = n.indicators.at(0);
    j["indicatorData"][n.name] = {
        {"name", indicator.name},
        {"mean", indicator.mean},
        {"source", indicator.source},
    };
  }
  return j.dump(indent);
}

string AnalysisGraph::serialize_to_json_string(bool verbose, bool compact) {

    nlohmann::json j;
    j["id"] = this->id;
    j["experiment_id"] = this->experiment_id;

    // This is an unordered_map:
    // concept name → Boost graph vertex id
    // Indicator assignment and observations are recorded according to this id.
    // Concept  ids are continuous between 0 and this->num_vertices()
    // Edges are also described based on concept ids.
    //j["concepts"] = this->name_to_vertex;

    // To go for a more compressed version of concepts where array index is the
    // concept id
    j["concepts"] = {};

    if (!verbose) {
        j["periods"] = {};
        j["has_min"] = {};
        j["min_val_obs"] = {};
        j["has_max"] = {};
        j["max_val_obs"] = {};
    }

    // Concept to indicator mapping. This is an array of array of objects
    // Outer array goes from 0 to this->num_vertices() - 1 and it is indexed by
    // concept id. Inner array keeps all the indicator data for a single
    // concept. Each indicator is represented by a single json object.
    // Indicator ids (iid) are used to index into observations.
    // Irrespective of the order at which we populate the outer array, the json
    // library takes care of storing them ordered by the concept id.
    j["conceptIndicators"] = {};
    for (int v = 0; v < this->num_vertices(); v++) {
        Node &n = (*this)[v];

        if (verbose) {
            j["concepts"].push_back(
                        {{"concept", n.name},
                         {"cid", this->name_to_vertex.at(n.name)},
                         {"period", n.period},
                         {"has_min", n.has_min},
                         {"min_val_obs", n.min_val_obs},
                         {"has_max", n.has_max},
                         {"max_val_obs", n.max_val_obs}});

            for (Indicator &ind : n.indicators) {
                j["conceptIndicators"][this->name_to_vertex.at(n.name)].push_back(
                //j["conceptIndicators"][n.name].push_back(
                            {
                                {"indicator", ind.name},
                                {"iid", n.nameToIndexMap.at(ind.name)},
                                {"source", ind.source},
                                {"func", ind.aggregation_method},
                                {"unit", ind.unit}
                            }
                        );
            }
        } else {
            // This is a more compressed way to store concept information where
            // array index keeps track of the concept id
            j["concepts"][this->name_to_vertex.at(n.name)] = n.name;
            j["periods"][this->name_to_vertex.at(n.name)] = n.period;
            j["has_min"][this->name_to_vertex.at(n.name)] = n.has_min;
            j["min_val_obs"][this->name_to_vertex.at(n.name)] = n.min_val_obs;
            j["has_max"][this->name_to_vertex.at(n.name)] = n.has_max;
            j["max_val_obs"][this->name_to_vertex.at(n.name)] = n.max_val_obs;

            for (Indicator &ind : n.indicators) {
                // This is a more compressed representation. We do not store iid
                // separately. iid is the index at which this indicator information
                // object is stored at.
                j["conceptIndicators"]
                [this->name_to_vertex.at(n.name)]
                [n.nameToIndexMap.at(ind.name)] =
                            {
                                {"indicator", ind.name},
                                {"source", ind.source},
                                {"func", ind.aggregation_method},
                                {"unit", ind.unit}
                            };
            }
        }
    }
    // To test how things get ordered in the json side
    // We do not need to insert nodes according to the order of their ids.
    // json object created orders them properly.
    /*
    j["conceptIndicators"][8].push_back(
                {
                    {"indicator", "eight"},
                    {"iid", 8},
                    {"source", "src"},
                    {"func", "func"},
                    {"unit", "mm"}
                }
            );
    j["conceptIndicators"][5].push_back(
                {
                    {"indicator", "five"},
                    {"iid", 5},
                    {"source", "src"},
                    {"func", "func"},
                    {"unit", "mm"}
                }
            );
    */
    // Serialize the edges.
    // Edges are identified by the source and target vertex ids.
    j["edges"] = {};
    for (const auto e : this->edges()) {
        const Node &source = (*this)[boost::source(e, this->graph)];
        const Node &target = (*this)[boost::target(e, this->graph)];

        int num_evidence = this->edge(e).evidence.size();
        Evidence evidence = Evidence(num_evidence);
        for (int evid = 0; evid < num_evidence; evid++) {
            const Statement &stmt = this->edge(e).evidence[evid];
            const Event &subj = stmt.subject;
            const Event &obj  = stmt.object;

            evidence[evid] = {{subj.adjective, subj.polarity, subj.concept_name},
                              {obj.adjective, obj.polarity, obj.concept_name}};
        }
        if (verbose) {
            j["edges"].push_back({{"source", source.name},
                                {"target", target.name},
                                {"kernels", compact ? vector<double>()
                                                    : this->edge(e).kde.dataset},
                                {"evidence", evidence},
                                {"thetas", this->edge(e).sampled_thetas},
                                {"log_prior_hist",
                                   compact ? vector<double>()
                                           : this->edge(e).kde.log_prior_hist},
                                {"n_bins", this->edge(e).kde.n_bins},
                                {"theta", this->edge(e).get_theta()},
                                {"is_frozen", this->edge(e).is_frozen()}});
        }
        else {
            // This is a more compressed version of edges. We do not utilize space
            // for key names. Maybe once we have debugged the whole process, we
            // might be able to go for this.
            j["edges"].push_back(make_tuple(name_to_vertex.at(source.name),
                                            name_to_vertex.at(target.name),
                                            evidence,
                                            this->edge(e).sampled_thetas,
                                            compact ? vector<double>()
                                                : this->edge(e).kde.dataset,
                                            compact ? vector<double>()
                                                : this->edge(e).kde.log_prior_hist,
                                            this->edge(e).kde.n_bins,
                                            this->edge(e).get_theta(),
                                            this->edge(e).is_frozen()));
        }
    }

    if (verbose) {
        j["start_year"] = this->training_range.first.first;
        j["start_month"] = this->training_range.first.second;
        j["end_year"] = this->training_range.second.first;
        j["end_month"] = this->training_range.second.second;
    } else {
        // This is a pair of pairs where the first pair is <start_year,
        // start_month> and the second pair is <end_year, end_month>
        j["training_range"] = this->training_range;
    }

    j["num_modeling_timesteps_per_one_observation_timestep"] = this->num_modeling_timesteps_per_one_observation_timestep;
    j["train_start_epoch"] = this->train_start_epoch;
    j["train_end_epoch"] = this->train_end_epoch;
    j["train_timesteps"] = this->n_timesteps;
    j["modeling_timestep_gaps"] = this->modeling_timestep_gaps;
    j["observation_timesteps_sorted"] = this->observation_timesteps_sorted;
    j["model_data_agg_level"] = this->model_data_agg_level;

    // This contains all the observations. Indexing goes by
    // [ timestep ][ concept ][ indicator ][ observation ]
    // Concept and indicator indexes are according to the concept and indicator
    // ids mentioned above.
    j["observations"] = this->observed_state_sequence;

    j["trained"] = this->trained;

    if (this->trained) {
        // Serialize the sampled transition matrices and initial latent states
        // along with on the parameters related to training.
        // For transition matrices, we only need to serialize odd column
        // positions on even rows (even, odd) as all the other positions remain
        // constant.
        // For initial letter states, we only need to serialize odd positions
        // as those are the sampled derivatives. All the even positions stay
        // constant at 1.

        j["res"] = this->res;
        j["kde_kernels"] = this->n_kde_kernels;
        j["continuous"] = this->continuous;
        j["data_heuristic"] = this->data_heuristic;
        j["causemos_call"] = this->causemos_call;
        j["MAP_sample_number"] = this->MAP_sample_number;
        j["log_likelihood_MAP"] = this->log_likelihood_MAP;
        j["head_node_model"] = this->head_node_model;

        j["log_likelihoods"] = compact ? vector<double>()
                                       : this->log_likelihoods;

        int num_verts = this->num_vertices();
        int num_els_per_mat = num_verts * num_verts;

        // Instead of serializing things as sequences of matrices and vectors
        // we flatten them into one single long vectors. Here we are setting
        // the last element of each vector to a dummy value just to make the
        // json library allocate all the memory required to store these
        // vectors. These dummy values get overwritten with actual data.
        j["matrices"][num_els_per_mat * this->res - 1] = 11111111111111;
        j["S0s"][num_verts * this->res - 1] = 11111111111111;

        for (int samp = 0; samp < this->res; samp++) {
            for (int row = 0; row < num_verts; row++) {
                j["S0s"][samp * num_verts + row] = this->initial_latent_state_collection[samp](row * 2 + 1);

                for (int col = 0; col < num_verts; col++) {
                    j["matrices"][samp * num_els_per_mat + row * num_verts + col] = this->transition_matrix_collection[samp](row * 2, col * 2 + 1);
                }
            }
        }

        if (this->head_node_model == HNM_FOURIER) {
            /*
             * Transition matrix
             *      b ............ b 0 ............. 0
             *      b ............ b 0 ............. 0
             *
             *      h ............ h c ............. c
             *      h ............ h c ............. c
             *      0 ............ 0 s s 0 ......... 0
             *      0 ............ 0 s s 0 ......... 0
             *      0 ................ 0 s s 0 ..... 0
             *      0 ................ 0 s s 0 ..... 0
             *
             *      0 .......................... 0 s s
             *      0 .......................... 0 s s
             *
             *      b - Body node rows
             *      h - Seasonal head node rows
             *      c - Fourier coefficient rows
             *      s - Sinusoidal generating rows
             *
             *      Rows come in pairs: first and second derivative rows.
             *
             *      Here we are not saving the block of b and h values. Those
             *      are the sampled transition matrices.
             *
             *      sinusoidal_rows = # sinusoidal generating rows
             *                                   = # Fourier coefficient columns
             *
             *      # coefficient rows per head node  = 2
             *      # coefficients per head node      = 2 * sinusoidal_rows
             *      # coefficients for all head nodes = 2 * sinusoidal_rows * # head nodes
             *
             *      max # non-zero values per sinusoidal row = 2
             *      max # non-zero sinusoidal values         = 2 * sinusoidal_rows
             *
             *      Total number of values to save
             *        = 2 * sinusoidal_rows * # head nodes + 2 * sinusoidal_rows
             *        = 2 * sinusoidal_rows * (# head nodes + 1)
             *
             * Initial state
             *      # rows per head node  = 2
             *      # for all head nodes = 2 * # head nodes
             *
             *      # non-zero values per frequency       = 1
             *      # non-zero values for all frequencies = sinusoidal_rows / 2
             *      (Since we are using 0 radians as the initial angle, the
             *      initial state for sine curves = sin(0) = 0. So we do not
             *      need to save that)
             *
             *      Total number of values to save
             *                          = sinusoidal_rows / 2 + 2 * # head nodes
             *
             */
            // The first diagonal block of sinusoidal_start_idx by
            // sinusoidal_start_idx contains the transition matrix portion that
            // describes the relationships between concepts.
            int sinusoidal_start_idx = 2 * num_verts;

            // The next diagonal block of sinusoidal_rows by sinusoidal_rows
            // contains the transition matrix part that generates sinusoidals
            // of different frequencies.
            int sinusoidal_rows = this->A_fourier_base.rows() -
                                                           sinusoidal_start_idx;

            vector<int> head_node_ids_sorted(this->head_nodes.begin(),
                                             this->head_nodes.end());
            sort(head_node_ids_sorted.begin(), head_node_ids_sorted.end());

            // Set the last element with a dummy value to allocate memory
            j["A_fourier_base"][sinusoidal_rows * 2 *
                                    (this->head_nodes.size() + 1) - 1] = 111111;
            j["s0_fourier"][sinusoidal_rows / 2 +
                                      2 * this->head_nodes.size() - 1] = 111111;
            j["head_node_ids"] = head_node_ids_sorted;
            j["sinusoidal_rows"] = sinusoidal_rows;

            int n_hn = head_node_ids_sorted.size();
            int fouri_val_idx = 0; // To keep track of the next index to insert

            // Save Fourier coefficients of seasonal head nodes
            for (int hn_idx = 0; hn_idx < n_hn; hn_idx++) {
                int hn_id = head_node_ids_sorted[hn_idx];
                int dot_row = 2 * hn_id;
                int dot_dot_row = dot_row + 1;

                // Save coefficients for derivative row and second derivative
                // row.
                for (int row: {dot_row, dot_dot_row}) {
                    // Save coefficients for one row
                    for (int col = 0; col < sinusoidal_rows; col++) {
                        j["A_fourier_base"][fouri_val_idx++] = this->A_fourier_base(row,
                                                 sinusoidal_start_idx + col);
                    }
                }

                // Saving initial value and derivative for head node hn_id
                // For the state vector, dot_row is the value row and
                // dot_dot_row is the dot row (We are just reusing the same
                // indexes)
                j["s0_fourier"][2 * hn_idx] = this->s0_fourier(dot_row);
                j["s0_fourier"][2 * hn_idx + 1] = this->s0_fourier(dot_dot_row);
            }

            // Saving different frequency sinusoidal curves generating portions
            for (int row = 0; row < sinusoidal_rows; row += 2) {
                int dot_row = sinusoidal_start_idx + row;

                if (this->continuous) {
                    // There is only two non-zero values for a pair of rows.
                    // this->A_fourier_base(dot_row, dot_row + 1) is always 1,
                    // and so we do not need to save it.
                    j["A_fourier_base"][fouri_val_idx++] =
                                this->A_fourier_base(dot_row + 1, dot_row);
                } else { /// Discrete
                    // Extract a 2 x 2 block of non-zero values.
                    for (int r = 0; r < 2; r++) {
                        for (int c = 0; c < 2; c++) {
                            j["A_fourier_base"][fouri_val_idx++] =
                                          this->A_fourier_base(dot_row + r,
                                                                   dot_row + c);
                        }
                    }
                }

                // Saving the initial state for cosine curves
                // For the state vector, dot_dot_row is the dot row (We are just
                // reusing the same index)
                j["s0_fourier"][2 * n_hn + row] = this->s0_fourier(dot_row + 1);
            }
        }
    }

    return j.dump(4);
}

void AnalysisGraph::export_create_model_json_string() {
  nlohmann::json j;
  j["id"] = this->id;
  j["experiment_id"] = this->experiment_id;
  j["statements"] = {};
  j["statements"].push_back({{"belief", 1}});
  j["conceptIndicators"] = {};
  cout << j.dump(4) << endl;
}
