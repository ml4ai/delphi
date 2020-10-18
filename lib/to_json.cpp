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

string AnalysisGraph::serialize_to_json_string(bool verbose) {
    nlohmann::json j;
    j["id"] = this->id;

    // This is an unordered_map:
    // concept name → Boost graph vertex id
    // Indicator assignment and observations are recorded according to this id.
    // Concept  ids are continuous between 0 and this->num_vertices()
    // Edges are also described based on concept ids.
    //j["concepts"] = this->name_to_vertex;

    // To go for a more compressed version of concepts where array index is the
    // concept id
    j["concepts"] = {};

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
                         {"cid", this->name_to_vertex.at(n.name)}});

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
            j["edges"].push_back({{"source",name_to_vertex.at(source.name)},
                                {"target", name_to_vertex.at(target.name)},
                                {"kernels", this->edge(e).kde.dataset},
                                {"evidence", evidence}});
        }
        else {
            // This is a more compressed version of edges. We do not utilize space
            // for key names. Maybe once we have debugged the whole process, we
            // might be able to go for this.
            j["edges"].push_back(make_tuple(name_to_vertex.at(source.name),
                                            name_to_vertex.at(target.name),
                                            this->edge(e).kde.dataset,
                                            evidence));
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

    // This contains all the observations. Indexing goes by
    // [ timestep ][ concept ][ indicator ][ observaton ]
    // Concept and indicator indexes are according to the concept and indicator
    // ids mentioned above.
    j["observations"] = this->observed_state_sequence;

    j["trained"] = this->trained;
    if (this->trained) {
        j["res"] = this->res;
        j["continuous"] = this->continuous;
        j["data_heuristic"] = this->data_heuristic;
        j["causemos_call"] = this->causemos_call;

        int num_verts = this->num_vertices();
        int num_els_per_mat = num_verts * num_verts;
        //int num_els = this->res * num_verts * num_verts;

        //vector<vector<double>> mats = vector<vector<double>> (res);
        //vector<vector<double>> derivs = vector<vector<double>> (res);
        //vector<double> mats = vector<double> (num_els_per_mat * this->res);
        //vector<double> derivs = vector<double> (num_verts * this->res);

        j["matrices"][num_els_per_mat * this->res - 1] = 11111111111111;
        j["S0s"][num_verts * this->res - 1] = 11111111111111;
        for (int samp = 0; samp < this->res; samp++) {
            //mats[samp] = vector<double> (num_verts * num_verts);
            //derivs[samp] = vector<double> (num_verts * num_verts);
            for (int row = 0; row < num_verts; row++) {
                //derivs[samp][row] = this->initial_latent_state_collection[samp](row * 2 + 1);
                //derivs[samp * num_verts + row] = this->initial_latent_state_collection[samp](row * 2 + 1);
                j["S0s"][samp * num_verts + row] = this->initial_latent_state_collection[samp](row * 2 + 1);

                for (int col = 0; col < num_verts; col++) {
                    //mats[samp][row * num_verts + col] = this->transition_matrix_collection[samp](row * 2, col * 2 + 1);
                    //mats[samp * num_els_per_mat + row * num_verts + col] = this->transition_matrix_collection[samp](row * 2, col * 2 + 1);
                    j["matrices"][samp * num_els_per_mat + row * num_verts + col] = this->transition_matrix_collection[samp](row * 2, col * 2 + 1);
                }
            }
        }

        //j["matrices"] = mats;
        // = derivs;
    }

    cout << j.dump(4) << endl;
    return j.dump(4);
}
