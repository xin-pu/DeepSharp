using DeepSharp.RL.Models;
using Action = DeepSharp.RL.Models.Action;

namespace DeepSharp.RL.Policies
{
    public interface IPolicy
    {
        /// <summary>
        ///     get next action according observation
        /// </summary>
        /// <param name="observation"></param>
        /// <returns></returns>
        public Action PredictAction(Observation reward);
    }
}