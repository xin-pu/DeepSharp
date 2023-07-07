using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;
using TorchSharp.Modules;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    internal class A2C : PolicyGradientAgengt
    {
        public A2C(Environ<Space, Space> env,
            int batchsize,
            float alpha = 0.01f,
            float beta = 0.01f,
            float gamma = 0.99f)
            : base(env, "ActorCritic")
        {
            Batchsize = batchsize;
            Gamma = gamma;
            Alpha = alpha;
            Beta = beta;
            /// Out is V[batchsize,1]
            Q = new Net(ObservationSize, 128, 1, DeviceType.CPU);
            ExpReplays = new EpisodeExpReplay(batchsize, gamma);

            var parameters = new[] {Q, PolicyNet}
                .SelectMany(a => a.parameters());
            Optimizer = Adam(parameters, Alpha);
        }

        /// <summary>
        ///     Episodes send to train
        /// </summary>
        public int Batchsize { protected set; get; }

        public float Alpha { protected set; get; }
        public float Beta { protected set; get; }
        public float Gamma { protected set; get; }
        public Module<torch.Tensor, torch.Tensor> Q { protected set; get; }
        public EpisodeExpReplay ExpReplays { protected set; get; }
        public Optimizer Optimizer { protected set; get; }

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
                ExpReplays.Enqueue(e);
            });

            var experienceCase = ExpReplays.All();
            var state = experienceCase.PreState;
            var action = experienceCase.Action;
            var valsRef = experienceCase.Reward;
            ExpReplays.Clear();

            Optimizer.zero_grad();

            var value = Q.forward(state);

            var lossValue = new MSELoss().forward(value, valsRef);
            lossValue.backward();

            var logProbV = torch.log(PolicyNet.forward(state)).gather(1, action);
            var logProbActionV = (valsRef - value.detach()) * logProbV;
            var lossPolicy = -logProbActionV.mean();


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