import BookmarkGraph from './components/BookmarkGraph'; // Now .tsx
import Search from './components/Search'; // Now .tsx

function App() {
  return (
    <main className="w-full h-screen flex flex-col items-center p-8 relative overflow-hidden bg-black">
      {/* Background Gradient Spot */}
      <div className="absolute top-[-20%] left-[50%] -translate-x-1/2 w-[800px] h-[600px] bg-blue-900/20 blur-[120px] rounded-full pointer-events-none z-0" />

      {/* Header */}
      <header className="w-full flex justify-center items-center relative z-10 max-w-6xl w-full mb-8">
        <h1 className="text-4xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-br from-white to-gray-400 text-center select-none">
          Start Page
        </h1>
      </header>
      
      {/* Search Bar - Always visible */}
      <div className="w-full max-w-xl z-20 mb-8">
        <Search />
      </div>
      
      {/* Main Content Area */}
      <div className="w-full flex-1 max-w-7xl animate-fade-in-up relative z-10 overflow-hidden flex flex-col">
           <div className="w-full h-full rounded-2xl border border-zinc-800 bg-zinc-900/30 backdrop-blur-sm overflow-hidden relative">
              <BookmarkGraph />
           </div>
      </div>
      
    </main>
  );
}

export default App;
