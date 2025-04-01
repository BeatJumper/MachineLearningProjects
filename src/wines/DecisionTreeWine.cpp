#include <iostream>
#include <vector>
#include <cmath>
#include <map>
#include <algorithm>
#include <set>
#include <fstream>
#include <sstream>
using namespace std;

struct DataPoint {
    vector<double> features;
    int label;
};

double calculateEntropy(const vector<DataPoint>& data) {
    map<int, int> labelCount;
    for (const auto& point : data) {
        labelCount[point.label]++;
    }
    double entropy = 0.0;
    for (const auto& pair : labelCount) {
        double p = (double)pair.second / data.size();
        entropy -= p * log2(p);
    }
    return entropy;
}

vector<vector<DataPoint>> splitData(const vector<DataPoint>& data, int featureIndex, double threshold) {
    vector<DataPoint> left, right;
    for (const auto& point : data) {
        if (point.features[featureIndex] <= threshold)
            left.push_back(point);
        else
            right.push_back(point);
    }
    return { left, right };
}

double calculateInformationGain(const vector<DataPoint>& data, int featureIndex) {
    double baseEntropy = calculateEntropy(data);
    set<double> uniqueValues;
    for (const auto& point : data) {
        uniqueValues.insert(point.features[featureIndex]);
    }
    double newEntropy = 0.0;
    for (double value : uniqueValues) {
        auto subsets = splitData(data, featureIndex, value);
        double prob = (double)subsets[0].size() / data.size();
        newEntropy += prob * calculateEntropy(subsets[0]);
        prob = (double)subsets[1].size() / data.size();
        newEntropy += prob * calculateEntropy(subsets[1]);
    }
    return baseEntropy - newEntropy;
}

int chooseBestFeature(const vector<DataPoint>& data, bool useGainRatio) {
    int bestFeature = -1;
    double bestMetric = 0.0;
    int numFeatures = data[0].features.size();

    for (int i = 0; i < numFeatures; ++i) {
        double gain = calculateInformationGain(data, i);
        double metric = gain;
        if (useGainRatio) {
            double splitInfo = 0.0;
            set<double> uniqueValues;
            for (const auto& point : data) {
                uniqueValues.insert(point.features[i]);
            }
            for (double value : uniqueValues) {
                auto subsets = splitData(data, i, value);
                double prob = (double)subsets[0].size() / data.size();
                splitInfo -= prob * log2(prob);
                prob = (double)subsets[1].size() / data.size();
                splitInfo -= prob * log2(prob);
            }
            metric = gain / splitInfo;
        }
        if (metric > bestMetric) {
            bestMetric = metric;
            bestFeature = i;
        }
    }
    return bestFeature;
}

int chooseBestFeatureID3(const vector<DataPoint>& data) {
    return chooseBestFeature(data, false);
}

int chooseBestFeatureC45(const vector<DataPoint>& data) {
    return chooseBestFeature(data, true);
}

int chooseBestFeatureCART(const vector<DataPoint>& data) {
    int bestFeature = -1;
    double bestGini = 1.0;
    int numFeatures = data[0].features.size();

    for (int i = 0; i < numFeatures; ++i) {
        set<double> uniqueValues;
        for (const auto& point : data) {
            uniqueValues.insert(point.features[i]);
        }
        for (double value : uniqueValues) {
            auto subsets = splitData(data, i, value);
            double gini = 0.0;
            for (const auto& subset : subsets) {
                if (subset.empty()) continue;
                double score = 0.0;
                map<int, int> labelCount;
                for (const auto& point : subset) {
                    labelCount[point.label]++;
                }
                for (const auto& pair : labelCount) {
                    double p = (double)pair.second / subset.size();
                    score += p * p;
                }
                gini += (1 - score) * subset.size() / data.size();
            }
            if (gini < bestGini) {
                bestGini = gini;
                bestFeature = i;
            }
        }
    }
    return bestFeature;
}

vector<DataPoint> loadDataset(const string& filename) {
    vector<DataPoint> dataset;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        DataPoint dataPoint;
        int label;
        ss >> label;
        dataPoint.label = label - 1; // ����Ϊ 0-based label
        double value;
        while (ss >> value) {
            dataPoint.features.push_back(value);
        }
        dataset.push_back(dataPoint);
    }
    return dataset;
}

int main() {
    string dataFile = "wine.data";
    vector<DataPoint> dataset = loadDataset(dataFile);

    int bestID3 = chooseBestFeatureID3(dataset);
    int bestC45 = chooseBestFeatureC45(dataset);
    int bestCART = chooseBestFeatureCART(dataset);

    cout << "ID3 best feature index: " << bestID3 << endl;
    cout << "C4.5 best feature index: " << bestC45 << endl;
    cout << "CART best feature index: " << bestCART << endl;

    getchar();
    getchar();
    
    return 0;
}