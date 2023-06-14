using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class DQN : Agent
    {
        public DQN(Environ<Space, Space> env) : base(env)
        {
        }

        public override Act PredictAction(Observation reward)
        {
            throw new NotImplementedException();
        }

        public override float Learn(Episode[] steps)
        {
            throw new NotImplementedException();
        }
    }
}