#ifndef FLASH_H
#define FLASH_H

#include <vector>
#include <string>
#include <cstring>  // for memcpy
#include <cstdint>
#include "log_macros.h"    
#include "FaceEmbedding.hpp"
#include "ospi_flash.h"

// Helper function to serialize FaceEmbeddingCollection
std::vector<uint8_t> Serialize(const FaceEmbeddingCollection &collection) {
    std::vector<uint8_t> buffer;

    // Serialize the number of embeddings
    uint32_t numPersons = collection.embeddings.size();
    buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&numPersons), reinterpret_cast<uint8_t*>(&numPersons) + sizeof(numPersons));

    for (const auto& face : collection.embeddings) {
        // Serialize the name
        uint32_t nameLength = face.name.size();
        buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&nameLength), reinterpret_cast<uint8_t*>(&nameLength) + sizeof(nameLength));
        buffer.insert(buffer.end(), face.name.begin(), face.name.end());     

        // Serialize the averageEmbedding
        for (double val : face.averageEmbedding) {
            buffer.insert(buffer.end(), reinterpret_cast<uint8_t*>(&val), reinterpret_cast<uint8_t*>(&val) + sizeof(val));
        }
    }

    return buffer;
}

// Helper function to deserialize FaceEmbeddingCollection
FaceEmbeddingCollection Deserialize(const std::vector<uint8_t> &buffer) {
    FaceEmbeddingCollection collection;
    size_t offset = 0;

    // Deserialize the number of embeddings
    uint32_t numPersons;

    std::memcpy(&numPersons, &buffer[offset], sizeof(numPersons));
    offset += sizeof(numPersons);

    for (uint32_t i = 0; i < numPersons; ++i) {
        FaceEmbedding face;

        // Deserialize the name
        uint32_t nameLength;
        std::memcpy(&nameLength, &buffer[offset], sizeof(nameLength));
        // printf("offset: %zu, nameLength: %u, buffer size: %zu\n", offset, nameLength, buffer.size());
        offset += sizeof(nameLength);
        face.name = std::string(buffer.begin() + offset, buffer.begin() + offset + nameLength);
        offset += nameLength;

        // Deserialize the averageEmbedding
        face.averageEmbedding.resize(64);
        for (double &val : face.averageEmbedding) {
            std::memcpy(&val, &buffer[offset], sizeof(val));
            offset += sizeof(val);
        }

        collection.embeddings.push_back(face);
    }

    return collection;
}

int32_t flash_send(const FaceEmbeddingCollection &data)
{
    int32_t ret;

    // Serialize the collection to raw bytes
    std::vector<uint8_t> serializedData = Serialize(data);

    // Prepare the write buffer
    uint16_t write_buff[2048] = {0};
    std::memcpy(write_buff, serializedData.data(), serializedData.size());

    // Write the serialized data to flash memory
    ret = ptrDrvFlash->ProgramData(0xC0000000, write_buff, serializedData.size());

    // Wait for flash operation to complete
    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
        info("busy \n");
    } while (flash_status.busy);

    return ret;
}

int32_t ospi_flash_read_collection(FaceEmbeddingCollection &collection)
{
    int32_t ret;
    uint16_t read_buff[2048];
    
    // Perform the read operation from flash memory
    ret = ptrDrvFlash->ReadData(0xC0000000, read_buff, sizeof(read_buff));  

    std::copy(read_buff + 4, read_buff + 2048, read_buff);

    // printf("Read new buf numPersons:\n");
    // for (size_t i = 520; i < 550; ++i) {
    //     printf("%02x ", read_buff[i]);
    // }
    // printf("\n");

    // Wait until the flash read operation is complete
    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
        info("busy \n");
    } while (flash_status.busy);

    // std::vector<uint8_t> serializedData(read_buff, read_buff + sizeof(read_buff));
    std::vector<uint8_t> serializedData;
    serializedData.reserve(4096);  // Reserve space for 1024 uint16_t entries
    for (uint16_t value : read_buff) {
        serializedData.push_back(static_cast<uint8_t>(value & 0xFF));
        serializedData.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    }

    // Deserialize the data into a FaceEmbeddingCollection object
    collection = Deserialize(serializedData);
 
    return ret; 
}

#endif 