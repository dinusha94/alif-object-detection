#ifndef FACE_EMBEDDING_H
#define FACE_EMBEDDING_H

#include <string>
#include <vector>
#include <cstdint>
#include <limits> 
#include "log_macros.h"
#include <cmath>

const size_t MAX_EMBEDDINGS_PER_PERSON = 5;  // Limit to 5 embeddings per person



// Struct to hold the embeddings for a single person
struct FaceEmbedding {
    std::string name;                              // Name of the person
    std::vector<std::vector<int8_t>> embeddings;   // Multiple int8 feature vectors
    std::vector<double> averageEmbedding;          // Average embedding for this person

    // Add a new embedding for this person, but limit the number to MAX_EMBEDDINGS_PER_PERSON
    void AddEmbedding(const std::vector<int8_t>& embedding) {
        if (embeddings.size() < MAX_EMBEDDINGS_PER_PERSON) {
            embeddings.push_back(embedding);
        } else {
            // You could log a message or handle the case when the limit is reached
            printf("Maximum embeddings reached for %s\n", name.c_str());
        }
    }
};

// Struct to hold the embeddings for multiple persons
struct FaceEmbeddingCollection {
    std::vector<FaceEmbedding> embeddings;

    // Add a new embedding for a person (multiple embeddings allowed)
    void AddEmbedding(const std::string& personName, const std::vector<int8_t>& faceEmbedding) {
        for (auto& embedding : embeddings) {
            if (embedding.name == personName) {
                embedding.AddEmbedding(faceEmbedding);
                return;
            }
        }
        // If person is not found, create a new entry with the first embedding
        FaceEmbedding newEmbedding{personName, {faceEmbedding}};
        embeddings.push_back(newEmbedding);
    }

    void AddAvgEmbedding(const std::string& personName, const std::vector<double>& avgEmbedding) {
        FaceEmbedding* personEmbedding = nullptr;
        for (auto& embedding : embeddings) {
            if (embedding.name == personName) {
                personEmbedding = &embedding;
                break;
            }
        }

        if (!personEmbedding) {
            // If person is not found, create a new entry with the first embedding
            FaceEmbedding newEmbedding;
            newEmbedding.name = personName;
            newEmbedding.averageEmbedding = avgEmbedding;
            embeddings.push_back(newEmbedding);
        }
        else{
            personEmbedding->averageEmbedding = avgEmbedding; 
        }

               
    }


    // Retrieve a face embedding by name
    const FaceEmbedding* GetEmbeddingByName(const std::string& personName) const {
        for (const auto& embedding : embeddings) {
            if (embedding.name == personName) {
                return &embedding;
            }
        }
        return nullptr; // Return null if not found
    }

    // Function to normalize a single embedding using Z-score normalization
    std::vector<double> normalizeEmbedding(const std::vector<int8_t>& embedding) const {
        size_t size = embedding.size();
        if (size == 0) return {};  // Return if empty

        // Calculate mean and standard deviation
        double sum = 0;
        for (int8_t val : embedding) {
            sum += val;
        }
        double mean = sum / static_cast<double>(size);

        double varianceSum = 0;
        for (int8_t val : embedding) {
            varianceSum += std::pow(val - mean, 2);
        }
        double stddev = std::sqrt(varianceSum / size);

        // Normalize the embedding
        std::vector<double> normalizedEmbedding(size);
        for (size_t i = 0; i < size; ++i) {
            normalizedEmbedding[i] = (embedding[i] - mean) / stddev;
        }

        return normalizedEmbedding;
    }

    std::vector<double> averageEmbeddings(const std::vector<std::vector<int8_t>>& embeddings) {
        if (embeddings.empty()) return {};

        size_t embeddingSize = embeddings[0].size();
        std::vector<double> averagedEmbedding(embeddingSize, 0.0);

        // Temporary vector for holding sums in double
        std::vector<double> tempSum(embeddingSize, 0.0);

        // Sum corresponding elements of all normalized embeddings
        for (const auto& embedding : embeddings) {
            std::vector<double> normalized = normalizeEmbedding(embedding);
            for (size_t i = 0; i < embeddingSize; ++i) {
                tempSum[i] += normalized[i];
            }
        }

        // Average and store as double
        for (size_t i = 0; i < embeddingSize; ++i) {
            averagedEmbedding[i] = tempSum[i] / embeddings.size();
        }

        return averagedEmbedding;
    }

    std::vector<double> CalculateAverageEmbeddingAndSave(const std::string& personName) {
        // Find the person's FaceEmbedding object
        FaceEmbedding* personEmbedding = nullptr;
        for (auto& embedding : embeddings) {
            if (embedding.name == personName) {
                personEmbedding = &embedding;
                break;
            }
        }

        if (!personEmbedding) {
            // If the person is not found, return an empty vector or handle it as needed
            printf("Person with name %s not found.\n", personName.c_str());
            return {};
        }

        // Get the embeddings for the person
        const std::vector<std::vector<int8_t>>& embeddingsForPerson = personEmbedding->embeddings;

        // Use the AverageEmbeddings function to calculate the average embedding
        std::vector<double> averageEmbedding = averageEmbeddings(embeddingsForPerson);

        // Save the average embedding for this person in the new variable
        personEmbedding->averageEmbedding = averageEmbedding;

        return averageEmbedding;
    }


    // Calculate Euclidean Distance between two vectors
    double CalculateEuclideanDistance(const std::vector<int8_t>& v1, const std::vector<int8_t>& v2) const {
        if (v1.size() != v2.size()) return std::numeric_limits<double>::infinity(); // Return a large value if sizes don't match

        double sum = 0.0;
        for (size_t i = 0; i < v1.size(); ++i) {
            sum += std::pow(static_cast<int>(v1[i]) - static_cast<int>(v2[i]), 2);
        }
        return std::sqrt(sum);
    }

    double CalculateCosineSimilarity(const std::vector<double>& v1, const std::vector<double>& v2) const {
        if (v1.size() != v2.size()) {
            return std::numeric_limits<double>::quiet_NaN(); // Return NaN if sizes don't match
        }

        double dotProduct = 0.0;
        double normV1 = 0.0;
        double normV2 = 0.0;

        // Compute dot product and norms (squared)
        for (size_t i = 0; i < v1.size(); ++i) {
            dotProduct += v1[i] * v2[i];
            normV1 += v1[i] * v1[i];
            normV2 += v2[i] * v2[i];
        }

        // Compute cosine similarity: dot product / (norm of v1 * norm of v2)
        double denominator = std::sqrt(normV1) * std::sqrt(normV2);
        if (denominator == 0.0) {
            return std::numeric_limits<double>::quiet_NaN(); // Return NaN if denominator is zero (to avoid division by zero)
        }

        return dotProduct / denominator;
    }

    std::string FindMostSimilarEmbedding(const std::vector<int8_t>& targetEmbedding) const {
        std::string mostSimilarPerson;
        double maxSimilarity = 0.5; // -std::numeric_limits<double>::infinity();  // Start with the lowest possible value for similarity

        std::vector<double> normalized_target = normalizeEmbedding(targetEmbedding);

        for (const auto& embedding : embeddings) {  // Iterate over persons
            // Use the stored average embedding for comparison
            if (!embedding.averageEmbedding.empty()) {
                double similarity = CalculateCosineSimilarity(normalized_target, embedding.averageEmbedding);  // Use Cosine Similarity

                if (similarity > maxSimilarity) {
                    maxSimilarity = similarity;  // Update the highest similarity value
                    mostSimilarPerson = embedding.name;  // Store the name of the most similar person
                }
            }
        }

        info("Most similar person: %s, Max similarity: %f \n", mostSimilarPerson.c_str(), maxSimilarity); 

        // if (maxSimilarity == -std::numeric_limits<double>::infinity()) {
        //     return "No similar embeddings found!";
        // }
        if (maxSimilarity > 0.6){
            return mostSimilarPerson;
        }
        else{
            return "identifying ...";
        }

        
    }

    // Function to print all embeddings in the collection
    void PrintEmbeddings() const {
        for (const auto& embedding : embeddings) {
            // Log the name of the person
            info("Name: %s\n", embedding.name.c_str());

            // Log each embedding
            for (size_t i = 0; i < embedding.embeddings.size(); ++i) {
                info("Embedding %zu: ", i + 1);
                
                // Log the int8 feature values of the embedding
                for (const auto& value : embedding.embeddings[i]) {
                    info("%d ", static_cast<int>(value));  // Cast to int to display as an integer
                }
                info("\n");  // End the line after printing the embedding
            }

            // Log the average embedding if available
        if (!embedding.averageEmbedding.empty()) {
            info("Average Embedding: ");
            for (const auto& value : embedding.averageEmbedding) {
                info("%f ", value);  // Cast to int to display as an integer
            }
            info("\n");  // End the line after printing the average embedding
        } else {
            info("No average embedding available.\n");
        }
            info("------------------------\n");  // Separator between different embeddings
        }
    }
    
};

#endif // FACE_EMBEDDING_H
