#include "AnalysisGraph.hpp"

ConfidenceIntervals AnalysisGraph::get_confidence_interval(Predictions preds) {
    ConfidenceIntervals cis;
    // The calculation of the 95% confidence interval about the median is
    // taken from:
    // https://www.ucl.ac.uk/child-health/short-courses-events/ \
    //     about-statistical-courses/research-methods-and-statistics/chapter-8-content-8
    int n = this->res;
    int lower_rank = int((n - 1.96 * sqrt(n)) / 2);
    int upper_rank = int((2 + n + 1.96 * sqrt(n)) / 2);
    int middle = n / 2;
    bool n_is_odd = (n % 2 == 1);

    lower_rank = lower_rank < 0 ? 0 : lower_rank;
    upper_rank = upper_rank >= n ? n-1 : upper_rank;

    for (auto [ind_name, ind_preds] : preds) {
        std::unordered_map<std::string, std::vector<double>> ind_cis;

        for (int ts = 0; ts < this->pred_timesteps; ts++) {

            ranges::sort(ind_preds[ts]);
            double median_value = n_is_odd ? ind_preds[ts][middle]
                : (ind_preds[ts][middle] + ind_preds[ts][middle - 1]) / 2.0;
            double lower_limit = ind_preds[ts][lower_rank];
            double upper_limit = ind_preds[ts][upper_rank];

            ind_cis["Median"].push_back(median_value);
            ind_cis["Upper 95% CI"].push_back(upper_limit);
            ind_cis["Lower 95% CI"].push_back(lower_limit);
        }
        cis[ind_name] = ind_cis;
    }
    return cis;
}

CompleteState AnalysisGraph::get_complete_state() {
    ConceptIndicators concept_indicators;
    Edges edges;
    Adjectives adjectives;
    Polarities polarities;
    Thetas thetas;
    Derivatives derivatives;
    Predictions predictions;
    Data data_set;

    for (int v : this->node_indices()) {
        for (auto [ind_name, vert] : (*this)[v].nameToIndexMap) {
            concept_indicators[(*this)[v].name].push_back(ind_name);
        }
    }

    for (auto e : this->edges()) {
        const Node& s = this->source(e);
        const Node& t = this->target(e);

        const Statement& stmt_0 = this->graph[e].evidence[0];
        const Event& subject = stmt_0.subject;
        const Event& object = stmt_0.object;

        std::string subj_adjective = subject.adjective;
        std::string obj_adjective = object.adjective;

        int subj_polarity = subject.polarity;
        int obj_polarity = object.polarity;

        edges.push_back(std::make_pair(s.name, t.name));
        adjectives.push_back(std::make_pair(subj_adjective, obj_adjective));
        polarities.push_back(std::make_pair(subj_polarity, obj_polarity));

        std::unordered_map<std::string, std::vector<double>> prior;
        std::unordered_map<std::string, std::vector<double>> sampled_thetas;
        prior["Prior"] = this->graph[e].kde.dataset;
        sampled_thetas["Sampled Thetas"] = this->graph[e].sampled_thetas;

        thetas.push_back(std::make_pair(this->graph[e].kde.dataset, this->graph[e].sampled_thetas));
    }

    for (auto [vert_name, vert_id] : this->name_to_vertex) {
        for (int samp = 0; samp < this->res; samp++) {
            derivatives[vert_name].push_back(this->initial_latent_state_collection[samp](vert_id * 2 + 1));
        }
    }
    /*
    int year = this->training_range.first.first;
    int month = this->training_range.first.second;
    int num_data_points =
        this->calculate_num_timesteps(this->training_range.first.first,
                this->training_range.first.second,
                this->training_range.second.first,
                this->training_range.second.second);
    std::vector<std::string> data_range;

    for (int t = 0; t < num_data_points ; t++) {
        data_range.push_back(std::to_string(year) + "-" + std::to_string(month));

        if (month == 12) {
            year++;
            month = 1;
        }
        else {
            month++;
        }
    }
    */
    for (auto [vert_name, vert_id] : this->name_to_vertex) {
        for (auto [ind_name, ind_id] : (*this)[vert_id].nameToIndexMap) {
            std::unordered_map<int, std::vector<double>> preds;
            std::unordered_map<std::string, std::vector<double>> data;

            for (int ts = 0; ts < this->pred_timesteps; ts++) {
                for (int samp = 0; samp < this->res; samp++) {
                    preds[ts].push_back(this-> 
                            predicted_observed_state_sequences[samp][ts][vert_id]
                            [ind_id]);
                }
            }
            predictions[ind_name] = preds;

            //for (int ts = 0; ts < num_data_points; ts++) {
            for (int ts = 0; ts < this->n_timesteps; ts++) {
                for (double obs : this->observed_state_sequence[ts][vert_id][ind_id]) {
                    data["Time Step"].push_back(ts);
                    data["Data"].push_back(obs);
                }
            }
            data_set[ind_name] = data;
        }
    }

    std::vector<long> data_range(this->n_timesteps);
    double timestep = 0;
    for (int ts = 0; ts < this->n_timesteps; ts++) {
        timestep += this->observation_timestep_gaps[ts];
        data_range[ts] = this->train_start_epoch +
                        timestep * this->modeling_period;
    }

    std::vector<double> prediction_range(this->pred_timesteps);

    for (int ts = 0; ts < this->pred_timesteps; ts++) {
      prediction_range[ts] = this->train_start_epoch + (this->pred_start_timestep + ts * this->delta_t) * this->modeling_period;
    }

    ConfidenceIntervals cis = get_confidence_interval(predictions);

    //return std::make_tuple(concept_indicators, edges, adjectives, polarities, thetas, derivatives, data_range, data_set, this->pred_range, predictions, cis);
    return std::make_tuple(concept_indicators, edges, adjectives, polarities, thetas, derivatives, data_range, data_set, prediction_range, predictions, cis);
}
