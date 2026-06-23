using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents.Deep.Value
{
	/// <summary>
	///     Double DQN (DDQN).
	///     Uses Q network to select actions and QTarget network to evaluate them,
	///     decoupling selection and evaluation to reduce overestimation.
	///     a* = argmax_a Q(s', a)
	///     y  = r + γ * QTarget(s', a*)
	/// </summary>
	public class DoubleDQN : DeepValueAgent
	{
		public DoubleDQN(Environ<Space, Space> env,
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
		///     N episodes collect experience → sample from replay buffer → Double DQN update.
		/// </summary>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();
			foreach (var _ in Enumerable.Range(0, N))
			{
				Environ.Reset();
				var epoch   = 0;
				var episode = new Episode();
				while (!Environ.IsComplete(epoch))
				{
					epoch++;
					var act  = GetEpsilonAct(Environ.Observation!.Value!);
					var step = Environ.Step(act, epoch);
					episode.Enqueue(step);

					Environ.CallBack?.Invoke(step);
					Environ.Observation = step.PostState;
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
		///     y = r + γ * QTarget(s', argmax_a Q(s', a))
		/// </summary>
		private float UpdateNet()
		{
			var batchSample = UniformExp.Sample(BatchSize);

			// Q(s, a) — current network estimate of chosen action
			var stateActionValue = Q.forward(batchSample.PreState)
				.gather(1, batchSample.Action).squeeze(-1);

			// Double DQN target:
			// a* = argmax_a Q(s', a)   ← use Q to select action
			// y  = r + γ * QTarget(s', a*)  ← use QTarget to evaluate it
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