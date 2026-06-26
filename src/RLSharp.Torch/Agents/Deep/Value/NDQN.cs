using RLSharp.Torch.Environs;
using RLSharp.Torch.ExpReplays;

namespace RLSharp.Torch.Agents.Deep.Value
{
	/// <summary>
	///     Noisy + Dueling + Double DQN (NDQN / Simplified Rainbow)
	///     Combines three DQN improvements:
	///     - Noisy: NoisyLinear parameterized exploration (no ε-greedy needed)
	///     - Dueling: V(s) + A(s,a) dual architecture
	///     - Double: Q selects action + QTarget evaluates, decoupling selection from evaluation
	/// </summary>
	public class NDQN : DeepValueAgent
	{
		public NDQN(EnvironmentBase<Space, Space> env,
			int                           n         = 1000,
			int                           c         = 10000,
			float                         gamma     = 0.99f,
			int                           batchSize = 32)
			: base(env, "NDQN")
		{
			C         = c;
			N         = n;
			BatchSize = batchSize;
			Gamma     = gamma;
			Epsilon   = 0f; // Noisy replaces ε-greedy

			Q       = new NoisyDuelingNet(ObservationSize, 128, ActionSize);
			QTarget = new NoisyDuelingNet(ObservationSize, 128, ActionSize);
			QTarget.load_state_dict(Q.state_dict());

			Optimizer  = SGD(Q.parameters(), 0.001);
			Loss       = MSELoss();
			UniformExp = new UniformExpReplay(C);
		}

		public float Gamma { get; }

		public int C { get; }

		public int N { get; }

		public int BatchSize { get; }

		public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

		public UniformExpReplay UniformExp { get; }

		private NoisyDuelingNet QNoisy => (NoisyDuelingNet)Q;

		private NoisyDuelingNet QTargetNoisy => (NoisyDuelingNet)QTarget;

		/// <summary>
		///     Learning loop:
		///     - ResetNoise before each episode (noise-based exploration)
		///     - Uses argmax policy (not ε-greedy)
		/// </summary>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();
			foreach (var _ in Enumerable.Range(0, N))
			{
				QNoisy.ResetNoise();

				EnvironmentBase.Reset();
				var epoch   = 0;
				var episode = new Episode();
				while (!EnvironmentBase.IsComplete(epoch))
				{
					epoch++;
					var ActionValue  = GetPolicyAct(EnvironmentBase.ObservationValue!.Value!);
					var step = EnvironmentBase.Step(ActionValue, epoch);
					episode.Enqueue(step);

					EnvironmentBase.CallBack?.Invoke(step);
					EnvironmentBase.ObservationValue = step.PostState;
				}

				learnOutCome.AppendStep(episode);
				UniformExp.Enqueue(episode);
				if (UniformExp.Buffers.Count >= C)
					learnOutCome.Evaluate = UpdateNet();
			}

			SyncTargetNetwork();
			return learnOutCome;
		}

		/// <summary>
		///     Double DQN style update + Noisy noise.
		///     y = r + γ * QTarget(s', argmax_a Q(s', a))
		/// </summary>
		private float UpdateNet()
		{
			QNoisy.ResetNoise();
			QTargetNoisy.ResetNoise();

			var batchSample = UniformExp.Sample(BatchSize);

			var stateActionValue = Q.forward(batchSample.PreState)
				.gather(1, batchSample.Action).squeeze(-1);

			// Double DQN: Q selects action, QTarget evaluates
			var bestActions = Q.forward(batchSample.PostState).argmax(1).unsqueeze(1);
			var nextStateValues = QTarget.forward(batchSample.PostState).gather(1, bestActions).squeeze(-1).detach();
			var expectedStateActionValue = batchSample.Reward + Gamma * nextStateValues;

			var loss = Loss.call(stateActionValue, expectedStateActionValue);

			Optimizer.zero_grad();
			loss.backward();
			Optimizer.step();
			return loss.item<float>();
		}

		private void SyncTargetNetwork()
		{
			var parameters = Q.state_dict();
			QTarget.load_state_dict(parameters);
		}
	}
}