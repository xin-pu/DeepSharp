using DeepSharp.FLWeb.Models;
using DeepSharp.RL.Environs;

namespace DeepSharp.FLWeb.Services;

/// <summary>
///     Extracts a GridState snapshot from FrozenLake + Step.
/// </summary>
public static class GridStateExtractor
{
    public static GridState Extract(FrozenLake env, Step step)
    {
        var cells = new List<GridCell>();
        for (var i = 0; i < env.LakeUnits.Count; i++)
        {
            var unit = env[i];
            cells.Add(new GridCell
            {
                Index = unit.Index,
                Row = unit.Row,
                Column = unit.Column,
                Role = unit.Role.ToString(),
                IsPlayer = unit.Index == env.PlayID
            });
        }

        // Action tensor: 0=Up, 1=Down, 2=Left, 3=Right
        var actionIndex = (int)step.Action.Value!.item<long>();

        return new GridState
        {
            Cells = cells,
            PlayerIndex = env.PlayID,
            Action = actionIndex,
            ActionName = actionIndex switch
            {
                0 => "Up",
                1 => "Down",
                2 => "Left",
                3 => "Right",
                _ => "?"
            },
            Reward = step.Reward.Value,
            IsComplete = step.IsComplete
        };
    }
}
