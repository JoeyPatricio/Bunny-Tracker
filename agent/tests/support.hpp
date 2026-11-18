// Fixture locations and loaders shared by the test files.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace bunnycam::test {

inline const std::filesystem::path kRepoDir = BUNNYCAM_REPO_DIR;
inline const std::filesystem::path kFixtures = kRepoDir / "agent" / "tests" / "fixtures";
inline const std::filesystem::path kModelsDir = kRepoDir / "server" / "models";
inline const std::filesystem::path kMotionFrames =
    kRepoDir / "server" / "tests" / "fixtures" / "motion_fixture_frames.npy";

inline std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

inline nlohmann::json read_json(const std::filesystem::path& path) {
    return nlohmann::json::parse(read_file(path));
}

inline std::vector<nlohmann::json> read_jsonl(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    std::vector<nlohmann::json> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) lines.push_back(nlohmann::json::parse(line));
    }
    return lines;
}

// Minimal .npy reader: version 1 or 2 headers, uint8 ('|u1'), C order only,
// which is all motion_fixture_frames.npy uses.
struct NpyU8 {
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> data;
};

inline NpyU8 read_npy_u8(const std::filesystem::path& path) {
    const std::string raw = read_file(path);
    if (raw.size() < 10 || raw.compare(0, 6, "\x93NUMPY") != 0) {
        throw std::runtime_error("not an npy file: " + path.string());
    }
    const int major = static_cast<unsigned char>(raw[6]);
    std::size_t header_len = 0;
    std::size_t offset = 0;
    if (major == 1) {
        header_len = static_cast<unsigned char>(raw[8]) | (static_cast<unsigned char>(raw[9]) << 8);
        offset = 10;
    } else {
        header_len = 0;
        for (int i = 3; i >= 0; --i)
            header_len = (header_len << 8) | static_cast<unsigned char>(raw[8 + i]);
        offset = 12;
    }
    const std::string header = raw.substr(offset, header_len);
    if (header.find("'|u1'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos) {
        throw std::runtime_error("npy reader handles uint8 C-order only: " + header);
    }
    NpyU8 out;
    const auto open = header.find('(', header.find("'shape'"));
    const auto close = header.find(')', open);
    std::stringstream dims(header.substr(open + 1, close - open - 1));
    for (std::string dim; std::getline(dims, dim, ',');) {
        if (dim.find_first_not_of(' ') != std::string::npos) out.shape.push_back(std::stoul(dim));
    }
    std::size_t count = 1;
    for (auto d : out.shape) count *= d;
    const std::size_t start = offset + header_len;
    if (raw.size() - start != count) throw std::runtime_error("npy size mismatch");
    out.data.assign(raw.begin() + static_cast<std::ptrdiff_t>(start), raw.end());
    return out;
}

}  // namespace bunnycam::test
