using DeepSharp.RL.Environs;
using DeepSharp.RL.ExperienceSources;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents.Deep.Value
{
	/// <summary>
	///     Cross-Entropy Guided Policy (CGP)
	///     Combines Q-learning with a policy network trained via cross-entropy.
	///     The Q-network is trained with standard TD learning (DQN-style).
	///     The policy network π(a|s) is trained with cross-entropy loss where the
	///     target distribution is the Boltzmann distribution derived from Q-values:
	///     d(s) = softmax(Q(s,·) / τ)
	///     L_π   = -Σ d(s) * log π(a|s)
	///     At inference, only the policy network is used (argmax of logits),
	///     avoiding the computational expense of evaluating Q-values for all actions.
	///     This method can be combined with most deep Q-learning variants and
	///     demonstrates improved stability across runs, hyperparameters, and tasks.
	///     Reference: "Cross-Entropy Guided Policy for Reinforcement Learning"
	/// </summary>
	public class CGP : DeepAgent
	{
		/// <summary>
		///     Create a Cross-Entropy Guided Policy agent.
		/// </summary>
		/// <param name="env">Environment.</param>
		/// <param name="n">Number of episodes per Learn call before syncing target network.</param>
		/// <param name="c">Capacity of experience replay buffer.</param>
		/// <param name="epsilon">Epsilon for ε-greedy exploration during training.</param>
		/// <param name="gamma">Discount factor.</param>
		/// <param name="batchSize">Batch size for training.</param>
		/// <param name="temperature">Temperature τ for Boltzmann target distribution from Q-values.</param>
		/// <param name="hiddenSize">Hidden layer size for Q-network and policy network.</param>
		/// <param name="qLr">Learning rate for Q-network optimizer.</param>
		/// <param name="policyLr">Learning rate for policy network optimizer.</param>
		public CGP(Environ<Space, Space> env,
			int                          n           = 1000,
			int                          c           = 10000,
			float                        epsilon     = 0.1f,
			float                        gamma       = 0.99f,
			int                          batchSize   = 32,
			float                        temperature = 1.0f,
			long                         hiddenSize  = 128,
			float                        qLr         = 0.001f,
			float                        policyLr    = 0.001f)
			: base(env, "CGP")
		{
			C           = c;
			N           = n;
			BatchSize   = batchSize;
			Epsilon     = epsilon;
			Gamma       = gamma;
			Temperature = temperature;

			// Q-network and target Q-network
			Q       = new Net(ObservationSize, hiddenSize, ActionSize);
			QTarget = new Net(ObservationSize, hiddenSize, ActionSize);
			QTarget.load_state_dict(Q.state_dict());

			// Policy network (outputs logits without softmax; softmax applied at inference)
			PolicyNet = new Net(ObservationSize, hiddenSize, ActionSize);

			// Separate optimizers for Q-network and policy network
			QOptimizer      = SGD(Q.parameters(), qLr);
			PolicyOptimizer = SGD(PolicyNet.parameters(), policyLr);

			// Q-network uses MSE loss for TD learning
			QLoss = MSELoss();
			// Policy network uses cross-entropy (used via functional API in UpdatePolicyNetwork)
			Loss = CrossEntropyLoss();

			UniformExp = new UniformExpReplay(C);
		}

		/// <summary>
		///     Discount factor γ.
		/// </summary>
		public float Gamma { get; }

		/// <summary>
		///     Experience replay buffer capacity.
		/// </summary>
		public int C { get; }

		/// <summary>
		///     Number of episodes per Learn call.
		/// </summary>
		public int N { get; }

		/// <summary>
		///     Batch size for training.
		/// </summary>
		public int BatchSize { get; }

		/// <summary>
		///     Temperature τ for Boltzmann target distribution.
		///     τ → 0: target is one-hot at argmax Q (greedy).
		///     τ → ∞: target is uniform.
		///     τ = 1: standard softmax.
		/// </summary>
		public float Temperature { get; }

		// ── Networks ──────────────────────────────────────────────────────

		/// <summary>
		///     Q-value network.
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> Q { get; protected set; }

		/// <summary>
		///     Target Q-network (periodically synced from Q).
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> QTarget { get; protected set; }

		/// <summary>
		///     Policy network π(a|s). Outputs logits; softmax applied at inference.
		/// </summary>
		public Module<torch.Tensor, torch.Tensor> PolicyNet { get; protected set; }

		// ── Optimizers and losses ──────────────────────────────────────────

		/// <summary>
		///     Optimizer for Q-network.
		/// </summary>
		public Optimizer QOptimizer { get; protected set; }

		/// <summary>
		///     Optimizer for policy network.
		/// </summary>
		public Optimizer PolicyOptimizer { get; protected set; }

		/// <summary>
		///     Loss function for Q-network (MSE for TD error).
		/// </summary>
		public Loss<torch.Tensor, torch.Tensor, torch.Tensor> QLoss { get; protected set; }

		// ── Experience replay ──────────────────────────────────────────────

		/// <summary>
		///     Uniform experience replay buffer.
		/// </summary>
		public UniformExpReplay UniformExp { get; }

		// ── DeepAgent overrides ────────────────────────────────────────────

		/// <inheritdoc />
		/// <remarks>
		///     MainNet returns the policy network — this is the network used at inference.
		/// </remarks>
		public override Module<torch.Tensor, torch.Tensor> MainNet => PolicyNet;

		/// <summary>
		///     Policy action: multinomial sampling from softmax probabilities.
		///     During training, this provides natural Boltzmann exploration.
		///     At inference, use GetGreedyAct for deterministic argmax selection
		///     to avoid the computational expense of sample-based policy.
		/// </summary>
		public override Act GetPolicyAct(torch.Tensor state)
		{
			var logits   = PolicyNet.forward(state.unsqueeze(0)).squeeze(0);
			var probs    = functional.softmax(logits, -1);
			var actIndex = torch.multinomial(probs, 1, true).ToInt32();
			return new Act(torch.from_array(new[] { actIndex }));
		}

		/// <summary>
		///     Greedy (argmax) action — deterministic, used at inference.
		///     Only one forward pass through the policy network, no Q-value computation.
		/// </summary>
		public Act GetGreedyAct(torch.Tensor state)
		{
			var logits       = PolicyNet.forward(state);
			var bestActIndex = torch.argmax(logits).ToInt32();
			var actTensor    = torch.from_array(new[] { bestActIndex });
			return new Act(actTensor);
		}

		/// <summary>
		///     Save both Q-network and policy network to a directory.
		/// </summary>
		public override void Save(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			Q.save(Path.Combine(dir, $"{name}_q.dat"));
			PolicyNet.save(Path.Combine(dir, $"{name}_policy.dat"));
		}

		/// <summary>
		///     Load both Q-network and policy network from a directory.
		/// </summary>
		public override void Load(string path)
		{
			var dir  = Path.GetDirectoryName(path) ?? ".";
			var name = Path.GetFileNameWithoutExtension(path);
			Q.load(Path.Combine(dir, $"{name}_q.dat"));
			PolicyNet.load(Path.Combine(dir, $"{name}_policy.dat"));
		}

		// ── Core training loop ─────────────────────────────────────────────

		/// <summary>
		///     CGP learning loop:
		///     1. Collect N episodes using multinomial sampling from policy (Boltzmann exploration).
		///     2. Store experience in replay buffer.
		///     3. When buffer is full, sample batch and update both Q-network and policy network.
		///     4. Sync target Q-network every N episodes.
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
					// Multinomial sampling from policy provides natural exploration
					var act  = GetPolicyAct(Environ.Observation!.Value!);
					var step = Environ.Step(act, epoch);
					episode.Enqueue(step);

					Environ.CallBack?.Invoke(step);
					Environ.Observation = step.PostState;
				}

				learnOutCome.AppendStep(episode);
				UniformExp.Enqueue(episode);
				if (UniformExp.Buffers.Count >= C)
					learnOutCome.Evaluate = UpdateNetworks();
			}

			// Sync Q-network weights to target Q-network
			SyncTargetNetwork();
			return learnOutCome;
		}

		// ── Network updates ────────────────────────────────────────────────

		/// <summary>
		///     Update both Q-network (TD learning) and policy network (cross-entropy guided by Q-values).
		/// </summary>
		/// <returns>Q-network loss value.</returns>
		private float UpdateNetworks()
		{
			var batchSample = UniformExp.Sample(BatchSize);

			// 1. Update Q-network with standard DQN TD learning
			var qLoss = UpdateQNetwork(batchSample);

			// 2. Update policy network with cross-entropy guided by Q-values
			UpdatePolicyNetwork(batchSample);

			return qLoss;
		}

		/// <summary>
		///     Standard DQN Q-network update:
		///     y   = r + γ * max_a' QTarget(s', a')
		///     L_Q = (y - Q(s, a))²
		///     Uses Double DQN style: action selection from Q, evaluation from QTarget.
		/// </summary>
		private float UpdateQNetwork(ExperienceCase batchSample)
		{
			// Q(s, a) — current network estimate for the chosen action
			var stateActionValue = Q.forward(batchSample.PreState)
				.gather(1, batchSample.Action).squeeze(-1);

			// Double DQN style target:
			// a*  = argmax_a Q(s', a)         — use Q to select action
			// y   = r + γ * QTarget(s', a*)    — use QTarget to evaluate it
			var bestActions = Q.forward(batchSample.PostState).argmax(1).unsqueeze(1);
			var nextStateValues = QTarget.forward(batchSample.PostState).gather(1, bestActions).squeeze(-1).detach();
			var expectedStateActionValue = batchSample.Reward + Gamma * nextStateValues;

			var loss = QLoss.call(stateActionValue, expectedStateActionValue);

			QOptimizer.zero_grad();
			loss.backward();
			QOptimizer.step();

			return loss.item<float>();
		}

		/// <summary>
		///     Policy network update via cross-entropy guided by Q-values:
		///     d(s)   = softmax(Q(s,·) / τ)     — target distribution (Boltzmann from Q)
		///     L_π    = -Σ d(s) * log π(a|s)    — cross-entropy loss
		///     The Q-network is detached (no grad) when computing the target distribution,
		///     so gradients only flow through the policy network.
		/// </summary>
		private float UpdatePolicyNetwork(ExperienceCase batchSample)
		{
			// Compute target distribution from Q-values (no gradient through Q)
			torch.Tensor targetDist;
			using (torch.no_grad())
			{
				var qValues = Q.forward(batchSample.PreState);              // [batch, actionSize]
				targetDist = functional.softmax(qValues / Temperature, -1); // [batch, actionSize]
			}

			// Policy network logits → log-softmax
			var policyLogits = PolicyNet.forward(batchSample.PreState); // [batch, actionSize]
			var logProbs     = functional.log_softmax(policyLogits, -1);

			// Cross-entropy: -Σ target * log(π)
			var loss = -(targetDist * logProbs).sum(1).mean();

			PolicyOptimizer.zero_grad();
			loss.backward();
			PolicyOptimizer.step();

			return loss.item<float>();
		}

		// ── Target network sync ────────────────────────────────────────────

		/// <summary>
		///     Sync Q-network weights to target Q-network.
		/// </summary>
		private void SyncTargetNetwork()
		{
			var parameters = Q.state_dict();
			QTarget.load_state_dict(parameters);
		}
	}
}