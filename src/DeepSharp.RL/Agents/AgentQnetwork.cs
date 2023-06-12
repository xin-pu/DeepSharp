using DeepSharp.RL.Environs;
using DeepSharp.RL.Models;

namespace DeepSharp.RL.Agents
{
    public class AgentQnetwork<T1, T2> : Agent<T1, T2>
        where T1 : Space
        where T2 : Space
    {
        public AgentQnetwork(Environ<T1, T2> env) : base(env)
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