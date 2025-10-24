import { useState, useRef, useEffect } from 'react';

export default function App() {
  const [inputValue, setInputValue] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  function handleSubmit() {
    const q = inputValue.trim();
    if (!q || isLoading) return;
    
    // Add user message
    const userMessage = { id: Date.now(), text: q, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);
    
    // Simulate AI response after a short delay
    setTimeout(() => {
      const aiMessage = { 
        id: Date.now() + 1, 
        text: `This is a simulated response to: "${q}"`, 
        sender: 'ai' 
      };
      setMessages(prev => [...prev, aiMessage]);
      setIsLoading(false);
    }, 2000);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }

  return (
    <div className="min-h-screen w-full bg-linear-to-br from-purple-50 via-pink-50 to-blue-50 flex items-center justify-center p-6">
      <div className="w-full max-w-4xl h-[90vh] mx-auto bg-white/85 backdrop-blur-sm rounded-3xl shadow-2xl flex flex-col">
        
        {/* Header */}
        <header className="flex items-center gap-3 p-6 border-b border-gray-200">
          <div
            className="w-12 h-12 bg-gray-900 rounded-2xl flex items-center justify-center shrink-0"
            aria-hidden="true"
          >
            <span className="text-white font-semibold" title="the warriors AI">W</span>
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-900">The Warriors AI</h1>
            <p className="text-xs text-gray-500">Here to assist you</p>
          </div>
        </header>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <p className="text-gray-600 text-sm mb-2">Hello future warrior</p>
              <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-3">
                The warriors are here to assist you
              </h2>
              <p className="text-gray-500 text-sm">
                Ask anything to get started
              </p>
            </div>
          ) : (
            <>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[70%] px-4 py-3 rounded-2xl ${
                      msg.sender === 'user'
                        ? 'bg-purple-600 text-white rounded-br-sm'
                        : 'bg-gray-100 text-gray-800 rounded-bl-sm'
                    }`}
                  >
                    <p className="text-sm">{msg.text}</p>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input Area */}
        <div className="p-6 border-t border-gray-200">
          <div className="relative w-full">
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5 pointer-events-none">
              <img
                src="/searchIcon.svg"
                alt="Search"
                className="w-full h-full object-contain block"
              />
            </div>

            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask whatever you want"
              className="w-full pl-12 pr-24 py-4 bg-white border border-gray-200 rounded-2xl text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-purple-300 focus:border-transparent transition-all disabled:opacity-50 disabled:cursor-not-allowed"
              autoComplete="off"
              disabled={isLoading}
            />

            <div className="absolute right-3 top-1/2 -translate-y-1/2">
              <button
                onClick={handleSubmit}
                className="inline-flex items-center px-4 py-2 rounded-xl text-sm font-semibold bg-purple-600 text-white hover:bg-purple-700 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-300 disabled:opacity-50 disabled:cursor-not-allowed"
                disabled={!inputValue.trim() || isLoading}
              >
                {isLoading ? 'Thinking...' : 'Ask'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}