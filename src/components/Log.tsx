//レスポンス速度を表示するコンポーネント

import React from 'react';

type LogProps  = {
    text: string;
}

export const Log: React.FC<LogProps> = ({ text }) => {
    return (
        <div>{text}</div>
    );
};
