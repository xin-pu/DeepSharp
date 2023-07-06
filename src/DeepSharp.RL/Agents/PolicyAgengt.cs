using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class PolicyAgengt : Agent
    {
        protected PolicyAgengt(Environ<Space, Space> env, string name)
            : base(env, name)
        {
        }
    }
}