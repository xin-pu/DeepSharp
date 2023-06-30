using DeepSharp.RL.Environs;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Deep Q Network
    ///     Now ObservationSpace use one-dimensional for test
    ///     Will support Multi in feature
    /// </summary>
    public class DQN : Agent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="c">update interval</param>
        /// <param name="n">Capacity of Experience pool</param>
        public DQN(Environ<Space, Space> env, int c, int n, float gamma = 0.9f)
            : base(env, "DQN")
        {
            ObservationSpace = (int) env.ObservationSpace!.N;
            ActionSpace = (int) env.ActionSpace!.N;
            C = c;
            N = n;
            Gamma = gamma;
            Net = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
            Optimizer = Adam(Net.parameters(), 0.01);
            Loss = CrossEntropyLoss();
            Experience = new ExperienceReplayBuffer(N);
        }

        public int ActionSpace { protected set; get; }
        public int ObservationSpace { protected set; get; }
        public float Gamma { protected set; get; }

        /// <summary>
        ///     update interval
        /// </summary>
        public int C { protected set; get; }

        /// <summary>
        ///     Capacity of Experience pool
        /// </summary>
        public int N { protected set; get; }

        public Net Net { protected set; get; }
        public Optimizer Optimizer { protected set; get; }
        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { protected set; get; }
        public ExperienceReplayBuffer Experience { protected set; get; }

        public override Act GetPolicyAct(torch.Tensor state)
        {
            return GetPredValues(state).Item1;
        }


        public Tuple<Act, torch.Tensor> GetPredValues(torch.Tensor state)
        {
            var values = Net.forward(state);
            var bestActIndex = torch.argmax(values).ToInt32();
            var actTensor = torch.from_array(new[] {bestActIndex});
            var act = new Act(actTensor);
            return new Tuple<Act, torch.Tensor>(act, values);
        }

        public Episode Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var (act, values) = GetPredValues(Environ.Observation!.Value!);
                var actIndex = act.Value!.ToInt32();
                var step = Environ.Step(act, epoch);

                Experience.Append(step);

                var expSample = Experience.Sample();
                var argMaxValue = GetPredValues(expSample.StateNew.Value!).Item2.data<float>().Max();
                var rr = expSample.Reward.Value;
                var y_true = expSample.IsComplete
                    ? rr
                    : rr + Gamma * argMaxValue;
                values[actIndex] = y_true;
                var r_pred = Net.forward(expSample.State.Value!);
                var output = Loss.call(r_pred, values);
                Optimizer.zero_grad();
                output.backward();

                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        public override Episode RunEpisode(PlayMode playMode = PlayMode.Agent)
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var act = playMode switch
                {
                    PlayMode.Sample => GetSampleAct(),
                    PlayMode.Agent => GetPolicyAct(Environ.Observation!.Value!),
                    PlayMode.EpsilonGreedy => GetEpsilonAct(Environ.Observation!.Value!),
                    _ => throw new ArgumentOutOfRangeException(nameof(playMode), playMode, null)
                };
                var step = Environ.Step(act, epoch);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }
    }


    public class ExperienceReplayBuffer
    {
        public ExperienceReplayBuffer(int N)
        {
            Capacity = N;
            Buffers = new Queue<Step>(N);
        }

        public int Capacity { protected set; get; }

        public void Append(Step step)
        {
            if (Buffers.Count == Capacity) Buffers.Dequeue();
            Buffers.Enqueue(step);
        }

        public void Append(IEnumerable<Step> steps)
        {
            steps.ToList().ForEach(Append);
        }

        public void Append(Episode episode)
        {
            Append(episode.Steps);
        }

        public Step Sample()
        {
            var length = Buffers.Count;
            var probs = torch.from_array(Enumerable.Repeat(1, length).ToArray(), torch.ScalarType.Float32);
            var randomIndex = torch.multinomial(probs, 1L).ToInt32();
            return Buffers.ElementAt(randomIndex);
        }

        public Queue<Step> Buffers { set; get; }
    }
}