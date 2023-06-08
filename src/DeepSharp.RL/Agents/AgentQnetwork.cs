using DeepSharp.RL.Environs;
using DeepSharp.RL.Models;

namespace DeepSharp.RL.Agents
{
    public class AgentQnetwork : Agent
    {
        public AgentQnetwork(Environ env) : base(env)
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