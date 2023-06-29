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

        public ValueTable ValueTable { protected set; get; }


        /// <summary>
        ///     ε 贪心策略
        ///     利用和探索 策略
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