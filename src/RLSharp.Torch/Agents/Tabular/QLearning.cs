using RLSharp.Torch.Environs;

namespace RLSharp.Torch.Agents.Tabular
{
	public class QLearning : TabularAgent
	{
		/// <summary>
		/// </summary>
		/// <param name="env"></param>
		/// <param name="epsilon">Epsilon of ε-greedy policy.</param>
		/// <param name="alpha">Learning rate.</param>
		/// <param name="gamma">Discount factor.</param>
		public QLearning(EnvironmentBase<Space, Space> env,
			float                                      epsilon = 0.1f,
			float                                      alpha   = 0.2f,
			float                                      gamma   = 0.9f) :
			base(env, "QLearning")
		{
			Epsilon = epsilon;
			Alpha   = alpha;
			Gamma   = gamma;
		}


		public float Alpha { get; protected set; }

		public float Gamma { get; protected set; }


		public override ActionValue GetPolicyAct(torch.Tensor state)
		{
			var action = QTable.GetBestAct(state);
			return action ?? GetSampleAct();
		}


		public override LearnOutcome Learn()
		{
			EnvironmentBase.Reset();
			var episode = new Episode();
			var epoch   = 0;
			while (!EnvironmentBase.IsComplete(epoch))
			{
				epoch++;
				var epsilonAct = GetEpsilonAct(EnvironmentBase.ObservationValue!.Value!);
				var step       = EnvironmentBase.Step(epsilonAct, epoch);

				Update(step);

				episode.Steps.Add(step);
				EnvironmentBase.CallBack?.Invoke(step);

				EnvironmentBase.ObservationValue = step.PostState; /// It's import for Update ObservationValue
			}

			var sumReward = episode.Steps.Sum(a => a.Reward.Value);
			episode.SumReward = new Reward(sumReward);

			return new LearnOutcome(episode);
		}

		public void Update(Step step)
		{
			var s     = step.PreState.Value!;
			var a     = step.Action.Value!;
			var r     = step.Reward.Value;
			var sNext = step.PostState.Value!;
			var q     = QTable[s, a];

			var aNext = GetPolicyAct(sNext);
			var qNext = QTable[sNext, aNext.Value!];

			QTable[s, a] = q + Alpha * (r + Gamma * qNext - q);
		}
	}
}