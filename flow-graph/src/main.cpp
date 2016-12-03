#include <tbb/flow_graph.h>

#include <iostream>
#include <vector>
#include <fstream>

#include <unistd.h>

using Matrix = std::vector<uint8_t>;
using MatrixWithPositions = std::pair<Matrix, std::vector<size_t>>;
using PositionsTuple = tbb::flow::tuple<MatrixWithPositions, MatrixWithPositions, MatrixWithPositions>;

auto constexpr MAX_MATRIX_COUNT = 100;
auto constexpr M = 500;
auto constexpr N = 700;

void usage(char const * name)
{
	std::cerr << "Usage: " << name << " OPTIONS" << std::endl;
	std::cerr << std::endl;
	std::cerr << "OPTIONS:" << std::endl;
	std::cerr << "  -b NUM      brightness value to search for [0..255] " << std::endl;
	std::cerr << "  -l LIMIT    max number of proccessing images at a time" << std::endl;
	std::cerr << "  -f FILE     log file" << std::endl;
}

std::vector<uint8_t> extractSquare(Matrix const & matrix, size_t pos)
{
	std::vector<uint8_t> square;
	int64_t p = pos;
	if (p - M - 1 > 0) square.push_back(matrix[p - M - 1]);
	if (p - M > 0) square.push_back(matrix[p - M]);
	if (p - M + 1 > 0) square.push_back(matrix[p - M + 1]);
	if (p - 1 > 0) square.push_back(matrix[p - 1]);
	if (p + 1 < matrix.size()) square.push_back(matrix[p + 1]);
	if (p + M - 1 < matrix.size()) square.push_back(matrix[p + M - 1]);
	if (p + M < matrix.size()) square.push_back(matrix[p + M]);
	if (p + M + 1 < matrix.size()) square.push_back(matrix[p + M + 1]);
	square.push_back(matrix[pos]);
	return square;
}

int main(int argc, char ** argv)
{
	if (argc < 3) {
		usage(argv[0]);
		return -1;
	}

	int br = -1;
	int maxImgCount = -1;
	std::string logFile;

	char opt;
	while (-1 != (opt = getopt(argc, argv, "b:l:f:"))) {
		switch (opt) {
			case 'b':
				br = atoi(optarg);
				break;
			case 'l':
				maxImgCount = atoi(optarg);
				break;
			case 'f':
				logFile = optarg;
				break;
			default:
				std::cerr << "unknown opt: " << opt << std::endl;
				return -1;
		}
	}

	if (br < 0 || br > 255) {
		usage(argv[0]);
		return -1;
	}
	uint8_t brightness = br;

	if (maxImgCount < 0) {
		usage(argv[0]);
		return -1;
	}

	using namespace tbb::flow;

	size_t generated = 0;
	graph g;

	srand(0);
	source_node<Matrix> source(g,
		[=, &generated] (Matrix & mat) {
			for (size_t i = 0; i < M * N; ++i)
				mat.push_back(rand());
			return ++generated < MAX_MATRIX_COUNT;
		}
	, false);

	limiter_node<Matrix> limiter(g, maxImgCount);
	make_edge(source, limiter);

	broadcast_node<Matrix> bcast1(g);
	make_edge(limiter, bcast1);

	function_node<Matrix, MatrixWithPositions> maxer(g, 1,
		[] (Matrix const & matrix) {
			std::vector<size_t> positions;
			uint8_t max = 0;
			for (size_t i = 0; i < matrix.size(); ++i) {
				if (matrix[i] == max)
					positions.push_back(i);
				else if (matrix[i] > max) {
					positions.clear();
					max = matrix[i];
					positions.push_back(i);
				}
			}
			return std::make_pair(matrix, positions);
		}
	);
	make_edge(bcast1, maxer);

	function_node<Matrix, MatrixWithPositions> eqer(g, 1,
		[brightness] (Matrix const & matrix) {
			std::vector<size_t> positions;
			for (size_t i = 0; i < matrix.size(); ++i)
				if (matrix[i] == brightness)
					positions.push_back(i);
			return std::make_pair(matrix, positions);
		}
	);
	make_edge(bcast1, eqer);

	function_node<Matrix, MatrixWithPositions> miner(g, 1,
		[] (Matrix const & matrix) {
			std::vector<size_t> positions;
			uint8_t min = 0;
			for (size_t i = 0; i < matrix.size(); ++i) {
				if (matrix[i] == min)
					positions.push_back(i);
				else if (matrix[i] < min) {
					positions.clear();
					min = matrix[i];
					positions.push_back(i);
				}
			}
			return std::make_pair(matrix, positions);
		}
	);
	make_edge(bcast1, miner);

	join_node<PositionsTuple> joiner(g);
	make_edge(maxer, input_port<0>(joiner));
	make_edge(eqer, input_port<1>(joiner));
	make_edge(miner, input_port<2>(joiner));

	broadcast_node<PositionsTuple> bcast2(g);
	make_edge(joiner, bcast2);

	function_node<PositionsTuple, bool> inversion(g, unlimited,
		[] (PositionsTuple const & t) {
			auto mwt = get<0>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				std::vector<uint8_t> inverted;
				for (auto j: v) inverted.push_back(255 - j);
				(void) inverted;
			}

			mwt = get<1>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				std::vector<uint8_t> inverted;
				for (auto j: v) inverted.push_back(255 - j);
				(void) inverted;
			}

			mwt = get<2>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				std::vector<uint8_t> inverted;
				for (auto j: v) inverted.push_back(255 - j);
				(void) inverted;
			}

			return true;
		}
	);
	make_edge(bcast2, inversion);

	function_node<PositionsTuple, bool> averageBrightness(g, unlimited,
		[&logFile] (PositionsTuple const & t) {
			uint64_t sum = 0;
			uint64_t count = 0;
			auto mwt = get<0>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				for (auto j: v) sum += j;
				count += v.size();
			}

			mwt = get<1>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				for (auto j: v) sum += j;
				count += v.size();
			}

			mwt = get<2>(t);
			for (auto i: mwt.second) {
				auto v = extractSquare(mwt.first, i);
				for (auto j: v) sum += j;
				count += v.size();
			}

			if (!logFile.empty()) {
				std::fstream log(logFile);
				log << sum / count;
			}

			return true;
		}
	);
	make_edge(bcast2, averageBrightness);

	join_node<tuple<bool, bool>> joiner2(g);
	make_edge(inversion, input_port<0>(joiner2));
	make_edge(averageBrightness, input_port<1>(joiner2));

	function_node<tuple<bool, bool>, continue_msg> decrement(g, unlimited,
		[] (tuple<bool, bool> const & t) {
			return continue_msg();
		}
	);
	make_edge(joiner2, decrement);
	make_edge(decrement, limiter.decrement);

	source.activate();
	g.wait_for_all();

	return 0;
}
