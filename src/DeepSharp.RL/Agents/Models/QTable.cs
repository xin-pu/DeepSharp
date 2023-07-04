using System.Text;
using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     State-Action Value Table
    ///     Q(s,a) = Cumulative reward
    /// </summary>
    public class QTable : IEquatable<QTable>
    {
        public QTable()
        {
            Return = new Dictionary<TransitKey, float>();
        }

        public Dictionary<TransitKey, float> Return { protected set; get; }
        protected List<TransitKey> TrasitKeys => Return.Keys.ToList();


        public float this[TransitKey transit]
        {
            get => GetValue(transit);
            set => SetValue(transit, value);
        }


        public float this[torch.Tensor state, torch.Tensor action]
        {
            get => GetValue(new TransitKey(state, action));
            set => SetValue(new TransitKey(state, action), value);
        }


        public bool Equals(QTable? other)
        {
            if (other == null) return false;
            if (other.TrasitKeys.Count != TrasitKeys.Count) return false;
            var res = TrasitKeys.All(key => !(Math.Abs(this[key] - other[key]) > 1E-2));
            return res;
        }


        private void SetValue(TransitKey transit, float value)
        {
            Return[transit] = value;
        }

        private float GetValue(TransitKey transit)
        {
            Return.TryAdd(transit, 0f);
            return Return[transit];
        }


        /// <summary>
        ///     argMax
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public Act? GetBestAct(torch.Tensor state)
        {
            var row = TrasitKeys
                .Where(a => a.State.Equals(state));

            var stateActions = Return
                .Where(a => row.Contains(a.Key)).ToList();

            if (!stateActions.Any())
                return null;

            if (stateActions.All(a => a.Value == 0))
                return null;

            var argMax = stateActions
                .MaxBy(a => a.Value);
            var act = argMax.Key.Act;
            return new Act(act);
        }

        /// <summary>
        ///     argMax
        /// </summary>
        /// <param name="state"></param>
        /// <returns></returns>
        public float GetBestValue(torch.Tensor state)
        {
            var row = TrasitKeys
                .Where(a => a.State.Equals(state));

            var stateActions = Return
                .Where(a => row.Contains(a.Key)).ToList();

            if (!stateActions.Any())
                return 0;

            var bestValue = stateActions
                .Max(a => a.Value);
            return bestValue;
        }

        public override string ToString()
        {
            var str = new StringBuilder();
            foreach (var keyValuePair in Return.Where(a => a.Value > 0))
                str.AppendLine($"{keyValuePair.Key}\t{keyValuePair.Value:F4}");
            return str.ToString();
        }
    }
}