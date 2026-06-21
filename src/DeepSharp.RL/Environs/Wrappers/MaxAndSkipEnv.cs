namespace DeepSharp.RL.Environs.Wrappers
{
	public abstract class MaxAndSkipEnv : EnvironWrapper
	{
		protected MaxAndSkipEnv(Environ<Space, Space> environ, int skip)
			: base(environ)
		{
			Skip         = skip;
			Observations = new Queue<Observation>(2);
		}

		public int Skip { get; protected set; }

		public Queue<Observation> Observations { get; protected set; }


		public override Step Step(Act act, int epoch)
		{
			var totalReward = 0f;
			var isComplete  = false;
			var oldobs      = Environ.Observation!;
			foreach (var _ in Enumerable.Range(0, Skip))
			{
				var step = Environ.Step(act, epoch);
				Observations.Enqueue(step.PostState);
				totalReward += step.Reward.Value;
				if (step.IsComplete)
				{
					isComplete = true;
					break;
				}
			}

			var obs    = Observations.Select(a => a.Value!).ToList();
			var max    = new Observation(torch.max(torch.vstack(obs)));
			var reward = new Reward(totalReward);
			return new Step(oldobs, act, max, reward, isComplete);
		}

		public override Observation Reset()
		{
			Observations.Clear();
			var obs = Environ.Reset();
			Observations.Enqueue(obs);
			return obs;
		}
	}
}