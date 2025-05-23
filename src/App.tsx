
import './App.css'
import Plot from './components/Plot';
import {FetchInitialData, FetchUpdateData} from './components/Fetch';
import DRZoom from './components/DRZoom';

function App() {
  return (
    <>
      <div className="grid grid-cols-4 grid-rows-8 gap-4 h-screen">
        {/* 左側のメニュー
        <div className="col-span-1 row-span-8">
          <h1 className="text-center text-2xl font-bold">Menu</h1>
          <ul className="list-disc list-inside">
            <li>Item 1</li>
            <li>Item 2</li>
            <li>Item 3</li>
          </ul>
        </div> */}
        {/* Plot画面 */}
        {/* <div className="col-span-4 row-span-6 bg-gray-200 p-4">
          <Plot fetchInitialDataPoints={() => FetchInitialData()} fetchDataPoints={FetchUpdateData}/>
        </div> */}
        <DRZoom/>
      </div>
    </>
  )
}

export default App
