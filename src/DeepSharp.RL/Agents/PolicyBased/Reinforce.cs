using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
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

            ExpReplays = new UniformExpReplay();
            Optimizer = Adam(PolicyNet.parameters(), Alpha);
        }

        /// <summary>
        ///     Episodes send to train
        /// </summary>
        public int Batchsize { protected set; get; }

        public float Gamma { protected set; get; }
        public float Alpha { protected set; get; }


        public Optimizer Optimizer { protected set; get; }

        public ExpReplay ExpReplays { protected set; get; }


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
            var qValues = torch.from_array(episodes.SelectMany(a => GetQValues(a)).ToArray()).view(-1, 1);


            var logProbV = torch.log(PolicyNet.forward(state)).gather(1, action);
            var logProbActionV = qValues * logProbV;
            var loss = -logProbActionV.mean();


            loss.backward();
            Optimizer.step();

            ExpReplays.Clear();
            return learnOutCome;
        }

        private float[] GetQValues(Episode episode)
        {
            var res = new List<float>();
            var sumR = 0f;
            var steps = episode.Steps;
            steps.Reverse();
            foreach (var s in steps)
            {
                sumR *= Gamma;
                sumR += s.Reward.Value;
                res.Add(sumR);
            }

            res.Reverse();
            return res.ToArray();
        }

        public LearnOutcome LearnEpisode()
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