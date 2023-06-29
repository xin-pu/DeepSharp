using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class QLearning : ValueAgent
    {
        public QLearning(Environ<Space, Space> env,
            float epsilon = 0.1f,
            float alpha = 0.2f,
            float gamma = 0.9f) :
            base(env)
        {
            Epsilon = epsilon;
            Alpha = alpha;
            Gamma = gamma;
        }


        public float Alpha { protected set; get; }
        public float Gamma { protected set; get; }


        public override Act GetPolicyAct(torch.Tensor state)
        {
            var action = ValueTable.GetBestAct(state);
            return action ?? GetSampleAct();
        }

        public override void Update(Episode episode)
        {
        }

        public void Update(Step step)
        {
            var state = step.State.Value!;
            var action = step.Action.Value!;
            var stateNew = step.StateNew.Value!;
            var reward = step.Reward.Value!;

            var currentTransit = new TransitKey(state, action);

            var bestValue = ValueTable.GetBestValue(stateNew);
            var newValue = reward + Gamma * bestValue;
            var oldValue = ValueTable[currentTransit];
            var finalValue = oldValue * (1 - Alpha) + newValue * Alpha;

            ValueTable[currentTransit] = finalValue;
        }

        public Episode Learn()
        {
            Environ.Reset();
            var episode = new Episode();
            var epoch = 0;
            while (Environ.IsComplete(epoch) == false)
            {
                epoch++;
                var act = GetEpsilonAct(Environ.Observation!.Value!);
                var step = Environ.Step(act, epoch);
                Update(step);
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
                Update(step);
                episode.Steps.Add(step);
                Environ.CallBack?.Invoke(step);
                Environ.Observation = step.StateNew; /// It's import for Update Observation
            }

            var orginalReward = episode.Steps.Sum(a => a.Reward.Value);
            var sumReward = orginalReward;
            episode.SumReward = new Reward(sumReward);
            return episode;
        }


        public override string ToString()
        {
            return "QLearning";
        }
    }

    public struct LearnRes
    {
        public float Loss { set; get; }
        public Episode Episode { set; get; }
    }
}