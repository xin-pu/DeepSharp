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
        public DQN(Environ<Space, Space> env, int c, int n, float epsilon = 0.1f, float gamma = 0.9f)
            : base(env, "DQN")
        {
            ObservationSpace = (int) env.ObservationSpace!.N;
            ActionSpace = (int) env.ActionSpace!.N;
            C = c;
            N = n;
            Epsilon = epsilon;
            Gamma = gamma;
            Net = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
            TGTNet = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
            TGTNet.load_state_dict(Net.state_dict());
            Optimizer = Adam(Net.parameters(), 0.0001);
            Loss = MSELoss();
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

        public Net TGTNet { protected set; get; }
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

        public Tuple<Act, torch.Tensor> GetTGTPredValues(torch.Tensor state)
        {
            var values = TGTNet.forward(state);
            var bestActIndex = torch.argmax(values).ToInt32();
            var actTensor = torch.from_array(new[] {bestActIndex});
            var act = new Act(actTensor);
            return new Tuple<Act, torch.Tensor>(act, values);
        }

        /// <summary>
        ///     Update Net after C
        /// </summary>
        public void Learn()
        {
            var epochTop = 0;
            while (epochTop++ <= C)
            {
                Environ.Reset();
                var episode = new Episode();
                var epoch = 0;
                while (Environ.IsComplete(epoch) == false)
                {
                    epoch++;

                    var act = GetSampleAct();
                    var step = Environ.Step(act, epoch);

                    episode.Steps.Add(step);
                    Environ.CallBack?.Invoke(step);
                    Environ.Observation = step.StateNew; /// It's import for Update Observation
                }

                Experience.Append(episode);
                UpdateNet();
            }

            TGTNet.load_state_dict(Net.state_dict());
        }

        private void UpdateNet()
        {
            foreach (var i in Enumerable.Range(0, 32))
            {
                /// Step 0 Get a random sample (ss,aa,rr,ss') for Experience
                var expSample = Experience.Sample();
                var state = expSample.State.Value!;
                var stateNext = expSample.StateNew.Value!;
                var reward = expSample.Reward.Value;


                /// Step 1 forward and Get Q(ss,aa)
                var (_, qValue) = GetPredValues(state);


                var (act, nextQValue) = GetTGTPredValues(stateNext);
                var nextStateValue = nextQValue.data<float>().Max();

                var exceptedStatrActionValues = reward +
                                                (expSample.IsComplete ? 0 : Gamma * nextStateValue);

                var actIndex = act.Value!.ToInt32();
                var valueArr = nextQValue.detach().data<float>().ToArray();
                valueArr[actIndex] = exceptedStatrActionValues;
                nextQValue = torch.from_array(valueArr);

                var output = Loss.call(qValue, nextQValue);

                Optimizer.zero_grad();
                output.backward();
                Optimizer.step();
            }
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

        public Step Sample(int batchSize)
        {
            var length = Buffers.Count;
            var probs = torch.from_array(Enumerable.Repeat(1, length).ToArray(), torch.ScalarType.Float32);
            var randomIndex = torch.multinomial(probs, 1L).ToInt32();
            return Buffers.ElementAt(randomIndex);
        }

        public Queue<Step> Buffers { set; get; }
    }
}