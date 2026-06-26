using RLSharp.Torch.Environs;
using RLSharp.Torch.ExpReplays;

namespace RLSharp.Torch.Agents.Deep.Value
{
	/// <summary>
	///     Double DQN (DDQN).
	///     Uses Q network to select actions and QTarget network to evaluate them,
	///     decoupling selection and evaluation to reduce overestimation.
	///     a* = argmax_a Q(s', a)
	///     y  = r + Î³ * QTarget(s', a*)
	/// </summary>
	public class DoubleDQN : DeepValueAgent
	{
		public DoubleDQN(EnvironmentBase<Space, Space> env,
			int                                n         = 1000,
			int                                c         = 10000,
			float                              epsilon   = 0.1f,
			float                              gamma     = 0.99f,
			int                                batchSize = 32)
			: base(env, "DoubleDQN")
		{
			C         = c;
			N         = n;
			BatchSize = batchSize;
			Epsilon   = epsilon;
			Gamma     = gamma;

			Q       = new Net(ObservationSize, 128, ActionSize);
			QTarget = new Net(ObservationSize, 128, ActionSize);
			QTarget.load_state_dict(Q.state_dict());

			Optimizer  = SGD(Q.parameters(), 0.001);
			Loss       = MSELoss();
			UniformExp = new UniformExpReplay(C);
		}

		public float Gamma { get; }

		/// <summary>
		///     Experience replay buffer capacity.
		/// </summary>
		public int C { get; }

		/// <summary>
		///     Update interval (sync target every N episodes).
		/// </summary>
		public int N { get; }

		/// <summary>
		///     Batch size for training.
		/// </summary>
		public int BatchSize { get; }

		/// <summary>
		///     Target network.
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

		/// <summary>
		///     Experience replay buffer.
		/// </summary>
		public UniformExpReplay UniformExp { get; }

		/// <summary>
		///     Double DQN learning loop.
		///     N episodes collect experience â†?sample from replay buffer â†?Double DQN update.
		/// </summary>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();
			foreach (var _ in Enumerable.Range(0, N))
			{
				EnvironmentBase.Reset();
				var epoch   = 0;
				var episode = new Episode();
				while (!EnvironmentBase.IsComplete(epoch))
				{
					epoch++;
					var ActionValue  = GetEpsilonAct(EnvironmentBase.ObservationValue!.Value!);
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
		///     Double DQN core update.
		///     y = r + Î³ * QTarget(s', argmax_a Q(s', a))
		/// </summary>
		private float UpdateNet()
		{
			var batchSample = UniformExp.Sample(BatchSize);

			// Q(s, a) â€?current network estimate of chosen action
			var stateActionValue = Q.forward(batchSample.PreState)
				.gather(1, batchSample.Action).squeeze(-1);

			// Double DQN target:
			// a* = argmax_a Q(s', a)   â†?use Q to select action
			// y  = r + Î³ * QTarget(s', a*)  â†?use QTarget to evaluate it
			var bestActions = Q.forward(batchSample.PostState).argmax(1).unsqueeze(1);
			var nextStateValues = QTarget.forward(batchSample.PostState).gather(1, bestActions).squeeze(-1).detach();
			var expectedStateActionValue = batchSample.Reward + Gamma * nextStateValues;

			var loss = Loss.call(stateActionValue, expectedStateActionValue);

			Optimizer.zero_grad();
			loss.backward();
			Optimizer.step();
			return loss.item<float>();
		}

		/// <summary>
		///     Sync Q network weights to QTarget.
		/// </summary>
		private void SyncTargetNetwork()
		{
			var parameters = Q.state_dict();
			QTarget.load_state_dict(parameters);
		}
	}
}