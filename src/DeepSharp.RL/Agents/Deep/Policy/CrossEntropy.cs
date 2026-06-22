using System.Diagnostics;
using DeepSharp.RL.ActionSelectors;
using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.Policy
{
	/// <summary>
	///     An Agent base on CrossEntropy Function
	///     Cross-Entropy Method
	///     http://people.smp.uq.edu.au/DirkKroese/ps/eormsCE.pdf
	/// </summary>
	public class CrossEntropy : DeepPolicyAgent
	{
		public CrossEntropy(Environ<Space, Space> environ,
			int                                   t,
			float                                 percentElite = 0.7f,
			int                                   hiddenSize   = 100) : base(environ, "CrossEntropy")
		{
			T            = t;
			PercentElite = percentElite;
			PolicyNet = new Net((int)environ.ObservationSpace!.N, hiddenSize, (int)environ.ActionSpace!.N,
				Device.type);
			Optimizer = Adam(PolicyNet.parameters(), 0.01);
			Loss      = CrossEntropyLoss();
		}

		public int T { get; protected set; }

		public float PercentElite { get; protected set; }


		/// <summary>
		///     Generate action probability distribution from observation, sample next action.
		/// </summary>
		/// <param name="observation"></param>
		/// <returns></returns>
		public override Act GetPolicyAct(torch.Tensor state)
		{
			var input       = state.unsqueeze(0);
			var sm          = Softmax(1);
			var actionProbs = sm.forward(PolicyNet.forward(input));
			var nextAction  = new ProbActionSelector().Select(actionProbs);
			var action      = new Act(nextAction);
			return action;
		}


		public override LearnOutcome Learn()
		{
			var episodes = RunEpisodes(T, PlayMode.Sample);
			var elite    = GetElite(episodes);

			var oars = elite.SelectMany(a => a.Steps)
				.ToList();

			var observations = oars
				.Select(a => a.PostState.Value)
				.ToList();
			var actions = oars
				.Select(a => a.Action.Value)
				.ToList();

			var observation = torch.vstack(observations!);
			var action      = torch.vstack(actions!);

			var loss = Learn(observation, action);

			return new LearnOutcome(episodes, loss);
		}

		
		/// <summary>
		///     Replace default Optimizer
		/// </summary>
		/// <param name="optimizer"></param>
		public void UpdateOptimizer(Optimizer optimizer)
		{
			Optimizer = optimizer;
		}

		/// <summary>
		///     Get Elite
		/// </summary>
		/// <param name="episodes"></param>
		/// <param name="percent"></param>
		/// <returns></returns>
		public virtual Episode[] GetElite(Episode[] episodes)
		{
			var reward = episodes
				.Select(a => a.SumReward.Value)
				.ToArray();
			var rewardP = reward.OrderByDescending(a => a)
				.Take((int)(reward.Length * PercentElite))
				.Min();

			var filterEpisodes = episodes
				.Where(e => e.SumReward.Value > rewardP)
				.ToArray();

			return filterEpisodes;
		}

		/// <summary>
		///     core function to update net
		/// </summary>
		/// <param name="observations">tensor from multi observations, size: [batch,observation size]</param>
		/// <param name="actions">tensor from multi actions, size: [batch,action size]</param>
		/// <returns>loss</returns>
		internal float Learn(torch.Tensor observations, torch.Tensor actions)
		{
			Debug.Assert(observations.shape.Last() == ObservationSize,
				$"Agent observations tensor should be [B,{ObservationSize}]");

			actions = actions.squeeze(-1);
			var pred   = PolicyNet.forward(observations);
			var output = Loss.call(pred, actions);

			Optimizer.zero_grad();
			output.backward();
			Optimizer.step();

			var loss = output.item<float>();
			return loss;
		}
	}
}