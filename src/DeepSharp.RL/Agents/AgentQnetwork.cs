using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public class AgentQnetwork : Agent
    {
        public AgentQnetwork(Environ<Space, Space> env) : base(env)
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