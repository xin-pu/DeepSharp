using DeepSharp.FLWeb.Models;
using DeepSharp.RL.Agents;
using DeepSharp.RL.Agents.Deep.ActorCritic;
using DeepSharp.RL.Agents.Deep.Policy;
using DeepSharp.RL.Agents.Deep.Value;
using DeepSharp.RL.Agents.Tabular;
using DeepSharp.RL.Environs;

namespace DeepSharp.FLWeb.Services
{
	/// <summary>
	///     Creates Agent instances based on TrainingConfig.
	/// </summary>
	public static class AgentFactory
	{
		public static Agent Create(TrainingConfig cfg, FrozenLake env)
		{
			return cfg.AgentType switch
			{
				"QLearning"           => new QLearning(env, cfg.Epsilon, cfg.Alpha, cfg.Gamma),
				"SARSA"               => new SARSA(env, cfg.Epsilon, cfg.Alpha, cfg.Gamma),
				"MonteCarloOnPolicy"  => new MonteCarloOnPolicy(env, cfg.Epsilon, cfg.T),
				"MonteCarloOffPolicy" => new MonteCarloOffPolicy(env, cfg.Epsilon, cfg.T),
				"DQN"                 => new DQN(env, cfg.N, cfg.C, cfg.Epsilon, cfg.Gamma, cfg.BatchSize),
				"REINFORCE"           => new Reinforce(env, cfg.BatchSize, cfg.Gamma, cfg.Alpha),
				"A2C"                 => new A2C(env, cfg.BatchSize, cfg.Alpha, cfg.Beta, cfg.Gamma),
				_                     => throw new ArgumentException($"Unknown agent type: {cfg.AgentType}")
			};
		}
	}
}