import type { ColumnDef } from "@tanstack/react-table"
import { DataTable } from "./DataTable"
import { Checkbox } from "@/components/ui/checkbox";

const dummyData: NearestPoint[] = [
    { index: 1, distance: 10, label: "Point A" },
    { index: 2, distance: 20, label: "Point B" },
    { index: 3, distance: 30, label: "Point C" },
]

export type NearestPoint = {
    index: number;
    distance: number;
    label: string;
}



export const columns: ColumnDef<NearestPoint>[] = [
    {
        id: "select",
        header: ({ table }) => (
        <Checkbox
            checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && "indeterminate")
            }
            onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
            aria-label="Select all"
        />
        ),
        cell: ({ row }) => (
        <Checkbox
            checked={row.getIsSelected()}
            onCheckedChange={(value) => row.toggleSelected(!!value)}
            aria-label="Select row"
        />
        ),
        enableSorting: false,
        enableHiding: false,
        },
        {
            accessorKey: "label",
            header: "Label",
        },
        {
            accessorKey: "distance",
            header: "Distance",
        },
]


export const NearestPointsList: React.FC<{ points?: NearestPoint[] }> = ({ points = dummyData }) => {
    return (
        <DataTable columns={columns} data={points} />
    )
}