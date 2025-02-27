import maze_env as gym
import algorithms.search as search
import algorithms.mdp as mdp
import benchmark as bench


def main():
    bench.benchmark_search("DFS", 20, search.depth_first_search, search.show_search_path, 15, False)
    bench.benchmark_search("BFS", 20, search.depth_first_search, search.show_search_path, 15, False)
    bench.benchmark_search("A*",  20, search.a_star,             search.show_search_path, 15, False)

    bench.benchmark_mdp("MDP Value Iteration", 20, mdp.mdp_value_iteration,  mdp.show_mdp_path, 15, False)
    bench.benchmark_mdp("MDP Value Iteration", 20, mdp.mdp_policy_iteration, mdp.show_mdp_path, 15, True)


if __name__ == "__main__":
    main()
