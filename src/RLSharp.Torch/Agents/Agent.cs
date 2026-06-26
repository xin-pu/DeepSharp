using RLSharp.Torch.Enumerates;
using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents
{
	/// <summary>
	///     Abstract agent ¡ª interacts with an environment via policy and learning.
	/// </summary>
	public abstract class Agent : IAgent
	{
		protected Agent(EnvironmentBase<Space, Space> env, string name)
		{
			Name    = name;
			EnvironmentBase = env;
			Device  = env.Device;
		}

		public long ObservationSize => EnvironmentBase.ObservationSpace!.N;

		public long ActionSize => EnvironmentBase.ActionSpace!.N;

		public float Epsilon { get; set; } = 0.2f;


		public string Name { get; protected set; }

		public torch.Device Device { get; protected set; }

		public EnvironmentBase<Space, Space> EnvironmentBase { get; protected set; }


		public abstract LearnOutcome Learn();

		public abstract void Save(string path);

		public abstract void Load(string path);

		/// <summary>
		///     Get an action according to the agent's policy ¦Ð(s).
		/// </summary>
		/// <param name="state">Current state.</param>
		/// <returns>Action chosen by the agent's policy.</returns>
		public abstract ActionValue GetPolicyAct(torch.Tensor state);


		/// <summary>
		///     Sample a random action from the action space.
		/// </summary>
		/// <returns></returns>
		public ActionValue GetSampleAct()
		{
			return EnvironmentBase.SampleAct();
		}


		/// <summary>
		///     Get an action by ¦Å-greedy: ¦Ð^¦Å(s).
		///     With probability ¦Å, pick a random action; otherwise use policy.
		/// </summary>
		/// <param name="state">Current state.</param>
		/// <returns></returns>
		public ActionValue GetEpsilonAct(torch.Tensor state)
		{
			var v = RandomProvider.NextDouble();
			var ActionValue = v < Epsilon
				? GetSampleAct()
				: GetPolicyAct(state);
			return ActionValue;
		}


		/// <summary>
		///     Run one complete episode using the specified play mode.
		/// </summary>
		/// <returns>The completed episode.</returns>
		public virtual Episode RunEpisode(
			PlayMode playMode = PlayMode.Agent)
		{
			EnvironmentBase.Reset();
			var episode = new Episode();
			var epoch   = 0;
			while (!EnvironmentBase.IsComplete(epoch))
			{
				epoch++;
				var ActionValue = playMode switch
				{
					PlayMode.Sample        => GetSampleAct(),
					PlayMode.Agent         => GetPolicyAct(EnvironmentBase.ObservationValue!.Value!),
					PlayMode.EpsilonGreedy => GetEpsilonAct(EnvironmentBase.ObservationValue!.Value!),
					_                      => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
				};
				var step = EnvironmentBase.Step(ActionValue, epoch);
				episode.Steps.Add(step);
				EnvironmentBase.CallBack?.Invoke(step);
				EnvironmentBase.ObservationValue = step.PostState; /// It's import for Update ObservationValue
			}

			var sumReward = EnvironmentBase.GetReturn(episode);
			episode.SumReward = new Reward(sumReward);
			return episode;
		}


		/// <summary>
		///     Run multiple complete episodes.
		/// </summary>
		/// <returns>Array of episodes.</returns>
		public virtual Episode[] RunEpisodes(int count,
			PlayMode                             playMode = PlayMode.Agent)
		{
			var episodes = new List<Episode>();
			foreach (var _ in Enumerable.Repeat(1, count))
				episodes.Add(RunEpisode(playMode));

			return episodes.ToArray();
		}

		/// <summary>
		/// </summary>
		/// <param name="testCount">test count</param>
		/// <returns>Average Reward</returns>
		public float TestEpisodes(int testCount)
		{
			var episode       = RunEpisodes(testCount);
			var averageReward = episode.Average(a => a.SumReward.Value);
			return averageReward;
		}


		public override string ToString()
		{
			return $"Agent[{Name}]";
		}
	}
}