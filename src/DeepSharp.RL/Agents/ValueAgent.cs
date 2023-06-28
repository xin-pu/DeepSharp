using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class ValueAgent : Agent
    {
        protected ValueAgent(Environ<Space, Space> env)
            : base(env)
        {
            QTable = new QTable();
        }

        public QTable QTable { set; get; }
    }
}