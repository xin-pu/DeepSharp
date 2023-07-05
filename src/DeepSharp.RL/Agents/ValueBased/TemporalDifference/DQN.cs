using DeepSharp.RL.Environs;
using DeepSharp.RL.ExperienceSources;
using static TorchSharp.torch.optim;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Deep Q Network
    ///     Now ObservationSpace use one-dimensional for test
    ///     Will support Multi in feature
    ///     Using TargetNet and Experience
    /// </summary>
    public class DQN : Agent
    {
        /// <summary>
        /// </summary>
        /// <param name="env"></param>
        /// <param name="n">update interval</param>
        /// <param name="c">Capacity of Experience pool</param>
        public DQN(Environ<Space, Space> env,
            int n = 1000,
            int c = 10000,
            float epsilon = 0.1f,
            float gamma = 0.99f,
            int batchSize = 32)
            : base(env, "DQN")
        {
            C = c;
            N = n;
            BatchSize = batchSize;
            Epsilon = epsilon;
            Gamma = gamma;


            ObservationSpace = (int) env.ObservationSpace!.N;
            ActionSpace = (int) env.ActionSpace!.N;

            Q = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
            QTarget = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
            QTarget.load_state_dict(Q.state_dict());
            Optimizer = SGD(Q.parameters(), 0.001);
            Loss = MSELoss();
            UniformExp = new UniformExpReplays(C);
        }

        public int ActionSpace { protected set; get; }
        public int ObservationSpace { protected set; get; }
        public float Gamma { protected set; get; }

        /// <summary>
        ///     Capacity of Experience pool
        /// </summary>
        public int C { protected set; get; }

        /// <summary>
        ///     Update interval
        /// </summary>
        public int N { protected set; get; }

        /// <summary>
        ///     Core Net
        /// </summary>
        public Net Q { protected set; get; }

        /// <summary>
        ///     Target Net
        /// </summary>
        public Net QTarget { protected set; get; }

        /// <summary>
        ///     Batch size of training
        /// </summary>
        public int BatchSize { protected set; get; }

        public Optimizer Optimizer { protected set; get; }

        public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { protected set; get; }

        public UniformExpReplays UniformExp { protected set; get; }


        public override Act GetPolicyAct(torch.Tensor state)
        {
            var values = Q.forward(state);
            var bestActIndex = torch.argmax(values).ToInt32();
            var actTensor = torch.from_array(new[] {bestActIndex});
            var act = new Act(actTensor);
            return act;
        }


        /// <summary>
        ///     Update Net after N
        /// </summary>
        public void Learn()
        {
            foreach (var _ in Enumerable.Range(0, N))
            {
                Environ.Reset();
                var epoch = 0;
                var episode = new Episode();
                while (Environ.IsComplete(epoch) == false)
                {
                    epoch++;
                    /// Step 2 ε greedy select an action
                    var act = GetEpsilonAct(Environ.Observation!.Value!);
                    /// Step 3 get reward and next state
                    var step = Environ.Step(act, epoch);
                    /// Step 4 save to Experience
                    episode.Enqueue(step);

                    Environ.CallBack?.Invoke(step);
                    Environ.Observation = step.PostState; /// It's import for Update Observation
                }

                /// Step 5 update Q from Experience
                UniformExp.Enqueue(episode);
                if (UniformExp.Buffers.Count >= C)
                    UpdateNet();
            }

            /// 每隔C次刚更新权重 Net -> TargetNet
            CopyQToTagget();
        }

        /// <summary>
        ///     Copy Q to QTarget
        /// </summary>
        private void CopyQToTagget()
        {
            var partmeters = Q.state_dict();
            QTarget.load_state_dict(partmeters);
        }

        /// <summary>
        ///     Update Q Parameter by gradient
        /// </summary>
        private float UpdateNet()
        {
            /// Get batch size Sample
            var batchSample = UniformExp.Sample(BatchSize);


            /// Calcluate => Q(a,s)
            var stateActionValue = Q.forward(batchSample.PreState).gather(1, batchSample.Action).squeeze(-1);

            /// Calcluate => y = r + γ*argmaxQ'(a,s)
            var nextStateValue = QTarget.forward(batchSample.PostState).max(1).values.detach();
            var expectedStatedActionValue = batchSample.Reward + Gamma * nextStateValue;

            /// Calcluate => Loss
            var loss = Loss.call(stateActionValue, expectedStatedActionValue);

            /// Backforward and Update Prameters
            Optimizer.zero_grad();
            loss.backward();
            Optimizer.step();
            return loss.item<float>();
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
                Environ.Observation = step.PostState; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }
    }
}