using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class ValueAgent : Agent
    {
        protected ValueAgent(Environ<Space, Space> env, string name)
            : base(env, name)
        {
            QTable = new QTable();
        }

        public QTable QTable { protected set; get; }


        /// <summary>
        ///     argmax(a') Q(state,a')
        ///     价值表中获取该状态State下最高价值的action'
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act GetPolicyAct(torch.Tensor state)
        {
            var action = QTable.GetBestAct(state);
            return action ?? GetSampleAct();
        }

        public override void Save(string path)
        {
            /// Save Q Table
        }

        public override void Load(string path)
        {
            /// Save Q Table
        }
    }
}