using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     Deep Q Network
    ///     Now ObservationSpace use one-dimensional for test
    ///     Will support Multi in feature
    /// </summary>
    public class DQN : Agent
    {
        public DQN(Environ<Space, Space> env, int c, int n)
            : base(env, "DQN")
        {
            ObservationSpace = (int) env.ObservationSpace!.N;
            ActionSpace = (int) env.ActionSpace!.N;
            C = c;
            N = n;
            Net = new Net(ObservationSpace, 128, ActionSpace, DeviceType.CPU);
        }

        public int ActionSpace { protected set; get; }
        public int ObservationSpace { protected set; get; }

        /// <summary>
        ///     update interval
        /// </summary>
        public int C { protected set; get; }

        /// <summary>
        ///     Capacity of Experience pool
        /// </summary>
        public int N { protected set; get; }

        public Net Net { set; get; }


        public override Act GetPolicyAct(torch.Tensor state)
        {
            var actionProb = Net.forward(state);
            var maxActIndex = torch.argmax(actionProb).ToInt32();
            var actTensor = torch.from_array(new[] {maxActIndex});
            return new Act(actTensor);
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
                //Update(step);
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
}