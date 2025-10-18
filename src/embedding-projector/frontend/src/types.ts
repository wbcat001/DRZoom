

export type ScatterPoint = {
    x: number;
    y: number;
    index: number;
    label?: string;
    color? : string;
}


export type NearestPoint = {
    index: number;
    distance: number;
    point: ScatterPoint;
}