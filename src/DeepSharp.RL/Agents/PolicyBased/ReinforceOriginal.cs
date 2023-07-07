using DeepSharp.RL.Environs;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     ReinforceOriginal: Learn after each episode
    /// </summary>
    public class ReinforceOriginal : PolicyGradientAgengt
    {
        public ReinforceOriginal(Environ<Space, Space> env,
            float gamma = 0.99f,
            float alpha = 0.01f)
            : base(env, "ReinforceOriginal")
        {
            Gamma = gamma;
            Alpha = alpha;
            Optimizer = Adam(PolicyNet.parameters(), Alpha);
        }

        /// <summary>
        ///     Episodes send to train
        /// </summary>
        public int Batchsize { protected set; get; }

        public float Gamma { protected set; get; }
        public float Alpha { protected set; get; }


        public Optimizer Optimizer { protected set; get; }


        public override LearnOutcome Learn()
        {
            var learnOutCome = new LearnOutcome();

            var episode = RunEpisode();
            var steps = episode.Steps;
            learnOutCome.AppendStep(episode.Steps);

            steps.Reverse();
            Optimizer.zero_grad();

            var g = 0f;

            foreach (var s in steps)
            {
                var reward = s.Reward.Value;
                var state = s.PreState.Value!.unsqueeze(0);
                var action = s.Action.Value!.view(-1, 1).to(torch.ScalarType.Int64);
                var logProb = torch.log(PolicyNet.forward(state)).gather(1, action);

                g = Gamma * g + reward;

                var loss = -logProb * g;
                loss.backward();
                learnOutCome.Evaluate = loss.item<float>();
            }

            Optimizer.step();

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