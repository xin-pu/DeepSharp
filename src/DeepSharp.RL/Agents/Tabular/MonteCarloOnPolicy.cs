using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Tabular
{
	/// <summary>
	///     Monte Carlo method — on-policy learning.
	/// </summary>
	public class MonteCarloOnPolicy : TabularAgent
	{
		/// <summary>
		/// </summary>
		/// <param name="env"></param>
		/// <param name="t">the leaning count of  each epoch </param>
		public MonteCarloOnPolicy(Environ<Space, Space> env, float epsilon = 0.1f, int t = 10)
			: base(env, "MonteCarloOnPolicy")
		{
			Epsilon = epsilon;
			T       = t;
			Count   = new Dictionary<TransitKey, int>();
		}

		public int T { get; protected set; }

		public Dictionary<TransitKey, int> Count { get; protected set; }


		public override LearnOutcome Learn()
		{
			Environ.Reset();
			var episode = new Episode();
			var epoch   = 0;
			var act     = GetEpsilonAct(Environ.Observation!.Value!);
			while (!Environ.IsComplete(epoch) && epoch < T)
			{
				epoch++;
				var step = Environ.Step(act, epoch);

				episode.Steps.Add(step);

				Environ.CallBack?.Invoke(step);
				Environ.Observation = step.PostState; /// It's import for Update Observation
			}

			Update(episode);

			var sumReward = episode.Steps.Sum(a => a.Reward.Value);
			episode.SumReward = new Reward(sumReward);


			var learnOut = new LearnOutcome(episode);

			return learnOut;
		}


		public void Update(Episode episode)
		{
			var lenth = episode.Length;
			var steps = episode.Steps;
			foreach (var t in Enumerable.Range(0, lenth))
			{
				var step  = steps[t];
				var key   = new TransitKey(step.PreState, step.Action);
				var r     = steps.Skip(t).Average(a => a.Reward.Value);
				var count = GetCount(key);
				QTable[key] = (QTable[key] * count + r) / (count + 1);
				SetCount(key, count                + 1);
			}
		}

		private int GetCount(TransitKey transitKey)
		{
			Count.TryAdd(transitKey, 0);
			return Count[transitKey];
		}

		private void SetCount(TransitKey transitKey, int value)
		{
			Count[transitKey] = value;
		}
	}
}