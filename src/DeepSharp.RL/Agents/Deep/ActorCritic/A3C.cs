using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep.ActorCritic
{
	/// <summary>
	///     A3C (Asynchronous Advantage Actor-Critic)
	///     Multi-worker parallel sampling, n-step TD to compute advantage, synchronous update of PolicyNet and ValueNet.
	///     In single-process scenarios, workers execute multiple episodes sequentially.
	///     Advantage = Σ γ^k * r_{t+k} + γ^n * V(s_{t+n}) - V(s_t)
	///     Loss = -log π(a|s) * Advantage + (Advantage)^2  (Policy gradient + Value MSE)
	/// </summary>
	public class A3C : ActorCriticAgent
	{
		public A3C(Environ<Space, Space> env,
			int                          workerCount = 4,
			float                        alpha       = 0.01f,
			float                        gamma       = 0.99f,
			int                          nStep       = 5)
			: base(env, "A3C")
		{
			WorkerCount = workerCount;
			Gamma       = gamma;
			Alpha       = alpha;
			NStep       = nStep;

			// PolicyNet is auto-created by ActorCriticAgent base class (using PGN)
			ValueNet = new Net(ObservationSize, 128, 1, DeviceType.CPU);

			var parameters = new[] { ValueNet, PolicyNet }
				.SelectMany(a => a.parameters());
			Optimizer = Adam(parameters, Alpha);
		}

		/// <summary>
		///     Number of parallel workers.
		/// </summary>
		public int WorkerCount { get; }

		/// <summary>
		///     Learning rate.
		/// </summary>
		public float Alpha { get; }

		/// <summary>
		///     Discount factor.
		/// </summary>
		public float Gamma { get; }

		/// <summary>
		///     N-step TD steps.
		/// </summary>
		public int NStep { get; }

		/// <summary>
		///     A3C learning loop:
		///     Each worker samples an episode → collect n-step returns → unified update.
		/// </summary>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();

			// 1. Worker sampling phase: collect multiple episodes in parallel
			var episodes = RunEpisodes(WorkerCount);

			// 2. Build training data: flatten each episode's steps into (s, a, n-step return)
			var states       = new List<torch.Tensor>();
			var actions      = new List<torch.Tensor>();
			var nStepReturns = new List<float>();

			foreach (var episode in episodes)
			{
				learnOutCome.AppendStep(episode);
				var steps      = episode.Steps;
				var stepsCount = steps.Count;

				for (var t = 0; t < stepsCount; t++)
				{
					// Compute n-step return: G_t = Σ_{k=0}^{n-1} γ^k * r_{t+k} + γ^n * V(s_{t+n})
					var nStepReturn = 0f;
					var discount    = 1f;
					var lastState   = torch.zeros(0);

					for (var k = 0; k < NStep && t + k < stepsCount; k++)
					{
						nStepReturn += discount * steps[t + k].Reward.Value;
						discount    *= Gamma;
						lastState   =  steps[t + k].PostState.Value!;
					}

					// If episode hasn't ended, bootstrap with ValueNet
					if (t + NStep < stepsCount)
					{
						var v = ValueNet.forward(lastState.unsqueeze(0)).item<float>();
						nStepReturn += discount * v;
					}

					states.Add(steps[t].PreState.Value!.unsqueeze(0));
					actions.Add(steps[t].Action.Value!.unsqueeze(0));
					nStepReturns.Add(nStepReturn);
				}
			}

			if (states.Count == 0)
				return learnOutCome;

			// 3. Batch processing
			var stateBatch  = torch.cat(states);
			var actionBatch = torch.cat(actions).to_type(torch.ScalarType.Int64);
			var returnBatch = torch.tensor(nStepReturns, torch.ScalarType.Float32).unsqueeze(-1);

			// 4. Update
			Optimizer.zero_grad();

			// Value loss: MSE(V(s), n-step return)
			var values    = ValueNet.forward(stateBatch);
			var lossValue = MSELoss().forward(values, returnBatch);
			lossValue.backward();

			// Policy loss: -log π(a|s) * Advantage
			var advantage  = returnBatch - values.detach();
			var logProbs   = torch.log(PolicyNet.forward(stateBatch)).gather(1, actionBatch);
			var lossPolicy = -(advantage * logProbs).mean();

			lossPolicy.backward();
			Optimizer.step();

			learnOutCome.Evaluate = lossPolicy.item<float>();
			return learnOutCome;
		}
	}
}