using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace DeepSharp.RL.Agents
{
	public class ActorCritic : PolicyGradientAgengt
	{
		public ActorCritic(Environ<Space, Space> env,
			int                                  batchsize,
			float                                alpha = 0.01f,
			float                                beta  = 0.01f,
			float                                gamma = 0.99f)
			: base(env, "ActorCritic")
		{
			Batchsize           = batchsize;
			Gamma               = gamma;
			Alpha               = alpha;
			Beta                = beta;
			Q                   = new Net(ObservationSize, 128, ActionSize, DeviceType.CPU);
			ExpReplays          = new EpisodeExpReplay(batchsize, gamma);
			ExpReplaysForPolicy = new EpisodeExpReplay(batchsize, gamma);
			var parameters = new[] { Q, PolicyNet }
				.SelectMany(a => a.parameters());
			Optimizer = Adam(parameters, Alpha);
		}

		/// <summary>
		///     Episodes send to train
		/// </summary>
		public int Batchsize { get; protected set; }

		public float Alpha { get; protected set; }

		public float Beta { get; protected set; }

		public float Gamma { get; protected set; }

		public Module<torch.Tensor, torch.Tensor> Q { get; protected set; }

		public EpisodeExpReplay ExpReplays { get; protected set; }

		public EpisodeExpReplay ExpReplaysForPolicy { get; protected set; }

		public Optimizer Optimizer { get; protected set; }

		/// <summary>
		///     QLearning for VNet
		/// </summary>
		/// <returns></returns>
		public override LearnOutcome Learn()
		{
			var learnOutCome = new LearnOutcome();

			var episodes = RunEpisodes(Batchsize);


			episodes.ToList().ForEach(e =>
			{
				learnOutCome.AppendStep(e);
				ExpReplays.Enqueue(e, false);
				ExpReplaysForPolicy.Enqueue(e);
			});

			var experienceCase = ExpReplays.All();
			var state          = experienceCase.PreState;
			var action         = experienceCase.Action;
			var reward         = experienceCase.Reward;
			var poststate      = experienceCase.PostState;
			ExpReplays.Clear();

			Optimizer.zero_grad();

			var stateActionValue           = Q.forward(state).gather(1, action);
			var nextStateValue             = Q.forward(poststate).max(1).values.detach();
			var expectedStatedActionValuey = reward + nextStateValue * Gamma;
			var lossValue                  = new MSELoss().forward(stateActionValue, expectedStatedActionValuey);
			lossValue.backward();

			var logProbV       = torch.log(PolicyNet.forward(state)).gather(1, action);
			var logProbActionV = stateActionValue.detach() * logProbV;
			var lossPolicy     = -logProbActionV.mean();


			lossPolicy.backward();
			Optimizer.step();


			learnOutCome.Evaluate = lossPolicy.item<float>();

			return learnOutCome;
		}


		public override void Save(string path)
		{
			if (File.Exists(path)) File.Delete(path);
			PolicyNet.save(path);
		}

		public override void Load(string path)
		{
			PolicyNet.load(path);
		}
	}
}