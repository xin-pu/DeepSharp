using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     State Action Value Table
    /// </summary>
    public class QTable
    {
        public QTable()
        {
            Return = new Dictionary<TrasitKey, float>();
        }

        public Dictionary<TrasitKey, float> Return { protected set; get; }
        protected List<TrasitKey> TrasitKeys => Return.Keys.ToList();

        public void Update(torch.Tensor state, torch.Tensor action, float value)
        {
            var existKey = TrasitKeys.Where(a => a.State.Equals(state) &&
                                                 a.Act.Equals(action)).ToList();
            if (existKey.Any())
                Return[existKey.First()] = value;
            else
                Return[new TrasitKey(state, action)] = value;
        }

        public float GetValue(torch.Tensor state, torch.Tensor action)
        {
            var existKey = TrasitKeys.Where(a => a.State.Equals(state) &&
                                                 a.Act.Equals(action)).ToList();

            var ret = existKey.Any()
                ? Return[existKey.First()]
                : 0;
            return ret;
        }

        /// <summary>
        ///     argMax
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public Act? GetArgMax(torch.Tensor state)
        {
            try
            {
                var row = TrasitKeys.Where(a => a.State.Equals(state));
                var stateActions = Return.Where(a => row.Contains(a.Key)).ToList();
                var argMax = stateActions.MaxBy(a => a.Value);
                var act = argMax.Key.Act;
                return new Act(act);
            }
            catch
            {
                return null;
            }
        }
    }
}