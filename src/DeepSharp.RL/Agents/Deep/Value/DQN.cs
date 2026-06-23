using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents.Deep.Value
{
	/// <summary>
	///     Deep Q Network.
	///     Currently uses one-dimensional observation space for testing.
	///     Uses TargetNet and Experience Replay.
	/// </summary>
	public class DQN : DeepValueAgent
	{
		/// <summary>
		/// </summary>
		/// <param name="env"></param>
		/// <param name="n">Update interval (sync target every N episodes).</param>
		/// <param name="c">Capacity of experience replay buffer.</param>
		public DQN(Environ<Space, Space> env,
			int                          n         = 1000,
			int                          c         = 10000,
			float                        epsilon   = 0.1f,
			float                        gamma     = 0.99f,
			int                          batchSize = 32)
			: base(env, "DQN")
		{
			C         = c;
			N         = n;
			BatchSize = batchSize;
			Epsilon   = epsilon;
			Gamma     = gamma;


			Q       = new Net(ObservationSize, 128, ActionSize, Device.type);
			QTarget = new Net(ObservationSize, 128, ActionSize, Device.type);
			QTarget.load_state_dict(Q.state_dict());
			Optimizer  = SGD(Q.parameters(), 0.001);
			Loss       = MSELoss();
			UniformExp = new UniformExpReplay(C);
		}


		public float Gamma { get; protected set; }

		/// <summary>
		///     Capacity of experience replay buffer.
		/// </summary>
		public int C { get; protected set; }

		/// <summary>
		///     Update interval for syncing target network.
		/// </summary>
		public int N { get; protected set; }

		/// <summary>
		///     Core Q network.
		/// </summary>
		/// <summary>
		///     Target network (periodically synced from Q).
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

		/// <summary>
		///     Batch size for training.
		/// </summary>
		public int BatchSize { get; protected set; }


		public UniformExpReplay UniformExp { get; protected set; }


		/// <summary>
		///     Update Net after N
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
					/// Step 2: ε-greedy select an action
					var act = GetEpsilonAct(Environ.Observation!.Value!);
					/// Step 3: get reward and next state
					var step = Environ.Step(act, epoch);
					/// Step 4: save to experience replay
					episode.Enqueue(step);

					Environ.CallBack?.Invoke(step);
					Environ.Observation = step.PostState; /// Important: update observation for next step
				}

				/// Step 5: update Q from experience
				learnOutCome.AppendStep(episode);
				UniformExp.Enqueue(episode);
				if (UniformExp.Buffers.Count >= C)
					learnOutCome.Evaluate = UpdateNet();
			}

			/// Every N episodes, sync Q weights to QTarget
			SyncTargetNetwork();

			return learnOutCome;
		}


		/// <summary>
		///     Sync Q network weights to QTarget.
		/// </summary>
		private void SyncTargetNetwork()
		{
			var partmeters = Q.state_dict();
			QTarget.load_state_dict(partmeters);
		}

		/// <summary>
		///     Update Q network parameters by gradient descent.
		/// </summary>
		private float UpdateNet()
		{
			// Get batch size samples
			using var batchSample = UniformExp.Sample(BatchSize);


			// Calculate => Q(s,a)
			var stateActionValue = Q.forward(batchSample.PreState).gather(1, batchSample.Action).squeeze(-1);

			// Calculate => y = r + γ*argmax Q'(a,s)
			using var nextStateValue = QTarget.forward(batchSample.PostState).max(1).values.detach();
			using var expectedStatedActionValue = CalculateTargets(
				batchSample.Reward,
				nextStateValue,
				batchSample.Done,
				Gamma);

			// Calculate => Loss
			var loss = Loss.call(stateActionValue, expectedStatedActionValue);

			// Backward pass and update parameters
			Optimizer.zero_grad();
			loss.backward();
			Optimizer.step();
			return loss.item<float>();
		}

		public static torch.Tensor CalculateTargets(
			torch.Tensor rewards,
			torch.Tensor nextStateValues,
			torch.Tensor done,
			float        gamma)
		{
			using var notDone = done.logical_not().to_type(rewards.dtype);
			return rewards + gamma * nextStateValues * notDone;
		}
	}
}