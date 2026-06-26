namespace RLSharp.Torch.Environs.Wrappers
{
	public abstract class MaxAndSkipEnv : EnvironWrapper
	{
		protected MaxAndSkipEnv(EnvironmentBase<Space, Space> environmentBase, int skip)
			: base(environmentBase)
		{
			Skip         = skip;
			Observations = new Queue<ObservationValue>(2);
		}

		public int Skip { get; protected set; }

		public Queue<ObservationValue> Observations { get; protected set; }


		public override Step Step(ActionValue actionValue, int epoch)
		{
			var totalReward = 0f;
			var isComplete  = false;
			var oldobs      = EnvironmentBase.ObservationValue!;
			foreach (var _ in Enumerable.Range(0, Skip))
			{
				var step = EnvironmentBase.Step(actionValue, epoch);
				Observations.Enqueue(step.PostState);
				totalReward += step.Reward.Value;
				if (step.IsComplete)
				{
					isComplete = true;
					break;
				}
			}

			var obs    = Observations.Select(a => a.Value!).ToList();
			var max    = new ObservationValue(torch.max(torch.vstack(obs)));
			var reward = new Reward(totalReward);
			return new Step(oldobs, actionValue, max, reward, isComplete);
		}

		public override ObservationValue Reset()
		{
			Observations.Clear();
			var obs = EnvironmentBase.Reset();
			Observations.Enqueue(obs);
			return obs;
		}
	}
}