using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    public abstract class ValueAgent : Agent
    {
        protected ValueAgent(Environ<Space, Space> env, string name)
            : base(env, name)
        {
            ValueTable = new ValueTable();
        }

        public ValueTable ValueTable { protected set; get; }

        public abstract Episode Learn();

        /// <summary>
        ///     argmax(a') Q(state,a')
        ///     价值表中获取该状态State下最高价值的action'
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public override Act GetPolicyAct(torch.Tensor state)
        {
            var action = ValueTable.GetBestAct(state);
            return action ?? GetSampleAct();
        }
    }
}