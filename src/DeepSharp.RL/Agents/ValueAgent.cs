using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class ValueAgent : Agent
    {
        protected ValueAgent(Environ<Space, Space> env)
            : base(env)
        {
            ValueTable = new ValueTable();
        }

        public ValueTable ValueTable { set; get; }
    }
}