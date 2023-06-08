using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Agents
{
    public class AgentQnetwork : Agent
    {
        public AgentQnetwork(Environ env) : base(env)
        {
        }

        public override Action PredictAction(Observation reward)
        {
            throw new NotImplementedException();
        }

        public override float Learn(Episode[] steps)
        {
            throw new NotImplementedException();
        }
    }
}