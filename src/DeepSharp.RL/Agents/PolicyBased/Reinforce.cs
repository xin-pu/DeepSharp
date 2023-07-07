using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     ReinforceOriginal: Learn by a batch episodes
    /// </summary>
    public class Reinforce : PolicyGradientAgengt
    {
        public Reinforce(Environ<Space, Space> env,
            int batchsize = 4,
            float gamma = 0.99f,
            float alpha = 0.01f)
            : base(env, "Reinforce")
        {
            Batchsize = batchsize;
            Gamma = gamma;
            Alpha = alpha;

            ExpReplays = new EpisodeExpReplay(batchsize, gamma);
            Optimizer = Adam(PolicyNet.parameters(), Alpha);
        }

        /// <summary>
        ///     Episodes send to train
        /// </summary>
        public int Batchsize { protected set; get; }

        public float Gamma { protected set; get; }
        public float Alpha { protected set; get; }


        public Optimizer Optimizer { protected set; get; }

        public EpisodeExpReplay ExpReplays { protected set; get; }


        public override LearnOutcome Learn()
        {
            var learnOutCome = new LearnOutcome();

            var episodes = RunEpisodes(Batchsize);

            Optimizer.zero_grad();

            episodes.ToList().ForEach(e =>
            {
                learnOutCome.AppendStep(e);
                ExpReplays.Enqueue(e);
            });

            var experienceCase = ExpReplays.All();
            var state = experienceCase.PreState;
            var action = experienceCase.Action;
            var qValues = experienceCase.Reward;
            ExpReplays.Clear();

            var logProbV = torch.log(PolicyNet.forward(state)).gather(1, action);
            var logProbActionV = qValues * logProbV;
            var loss = -logProbActionV.mean();


            loss.backward();
            Optimizer.step();


            learnOutCome.Evaluate = loss.item<float>();

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