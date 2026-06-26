using RLSharp.Torch.Agents;
using RLSharp.Torch.Agents.Deep.ActorCritic;
using RLSharp.Torch.Agents.Deep.Policy;
using RLSharp.Torch.Agents.Deep.Value;
using RLSharp.Torch.Agents.Tabular;
using RLSharp.Torch.Environs;
using RLSharp.Web.Models;

namespace RLSharp.Web.Services
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
				"PPO"                 => new PPO(env, cfg.BatchSize, cfg.Gamma, cfg.Alpha),
				_                     => throw new ArgumentException($"Unknown agent type: {cfg.AgentType}")
			};
		}
	}
}