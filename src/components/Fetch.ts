interface DataPoint {
    index: number;
    x: number;
    y: number;
  }

export const FetchInitialData = async (): Promise<DataPoint[]> => {
    const response = await fetch('http://localhost:8000/init', {
        method: 'GET',
        headers: {
        'Content-Type': 'application/json',
        },
        body: JSON.stringify({ options: 'test' }),
    });
    const data = await response.json();

    return ConvertData(data);
}

export const FetchUpdateData = async (indexes: number[]) => {  
    const response = await fetch('http://localhost:8000/zoom', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filter: indexes }),
    });
    const data = await response.json();
    return ConvertData(data);
}

const ConvertData = (data: any): DataPoint[] => {
    return data.map((d: any) => ({ index: d.index, x: d.data[0], y: d.data[1] }));
}