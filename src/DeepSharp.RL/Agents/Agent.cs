using DeepSharp.RL.Enumerates;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
	/// <summary>
	///     Abstract agent — interacts with an environment via policy and learning.
	/// </summary>
	public abstract class Agent : IAgent
	{
		protected Agent(Environ<Space, Space> env, string name)
		{
			Name    = name;
			Environ = env;
			Device  = env.Device;
		}

		public long ObservationSize => Environ.ObservationSpace!.N;

		public long ActionSize => Environ.ActionSpace!.N;

		public float Epsilon { get; set; } = 0.2f;


		public string Name { get; protected set; }

		public torch.Device Device { get; protected set; }

		public Environ<Space, Space> Environ { get; protected set; }


		public abstract LearnOutcome Learn();

		public abstract void Save(string path);

		public abstract void Load(string path);

		/// <summary>
		///     Get an action according to the agent's policy π(s).
		/// </summary>
		/// <param name="state">Current state.</param>
		/// <returns>Action chosen by the agent's policy.</returns>
		public abstract Act GetPolicyAct(torch.Tensor state);


		/// <summary>
		///     Sample a random action from the action space.
		/// </summary>
		/// <returns></returns>
		public Act GetSampleAct()
		{
			return Environ.SampleAct();
		}


		/// <summary>
		///     Get an action by ε-greedy: π^ε(s).
		///     With probability ε, pick a random action; otherwise use policy.
		/// </summary>
		/// <param name="state">Current state.</param>
		/// <returns></returns>
		public Act GetEpsilonAct(torch.Tensor state)
		{
			var d = new Random();
			var v = d.NextDouble();
			var act = v < Epsilon
				? GetSampleAct()
				: GetPolicyAct(state);
			return act;
		}


		/// <summary>
		///     Run one complete episode using the specified play mode.
		/// </summary>
		/// <returns>The completed episode.</returns>
		public virtual Episode RunEpisode(
			PlayMode playMode = PlayMode.Agent)
		{
			Environ.Reset();
			var episode = new Episode();
			var epoch   = 0;
			while (!Environ.IsComplete(epoch))
			{
				epoch++;
				var act = playMode switch
				{
					PlayMode.Sample        => GetSampleAct(),
					PlayMode.Agent         => GetPolicyAct(Environ.Observation!.Value!),
					PlayMode.EpsilonGreedy => GetEpsilonAct(Environ.Observation!.Value!),
					_                      => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
				};
				var step = Environ.Step(act, epoch);
				episode.Steps.Add(step);
				Environ.CallBack?.Invoke(step);
				Environ.Observation = step.PostState; /// It's import for Update Observation
			}

			var sumReward = Environ.GetReturn(episode);
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